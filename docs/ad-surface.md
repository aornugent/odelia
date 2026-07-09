# odelia's AD surface: exact gradients of an ODE run

**Scope:** `traitecoevo/odelia` — the reverse-mode automatic-differentiation surface:
what a gradient is made of, who owns what, the contract a System implements to be
differentiable, and how to build one. Self-contained.

**Companion:** [`ARCHITECTURE.md`](../ARCHITECTURE.md) — how the XAD `Tape` runtime is
compiled once and linked across the DLL boundary. Read it before touching the build;
this document is about the API, not the link.

---

## The one-sentence design

odelia differentiates **any reduction of an ODE run** — on one scalar-templated
`Solver`, with reverse-mode XAD compiled once — and only `double` ever crosses back to R.

Everything below follows: the System is templated on its scalar so the same code
simulates (`double`) and differentiates (`active`); the thing differentiated is a
caller's functional, not a built-in loss; adaptive numerics are recorded once and
replayed on fixed nodes so the tape carries no adaptive branching; and the active
machinery is born, used, and destroyed inside one C++ call.

---

## Ownership

R holds one object — the ordinary (double) `Solver`. Everything active hangs off it and
never surfaces:

```
R  ── holds ──▶  d : Solver<System<double>>          the double solver
                 │
                 ├─ System<double>                    immutable after the adaptive pass
                 ├─ recording                          produced once, read per call:
                 │    ├─ L1  recorded_steps()          the discovered schedule   (Solver state)
                 │    ├─ L2  positions_history[step]   adaptive node positions   (System state)
                 │    └─ L3  field_history[step][stage] frozen field values      (System state)
                 └─ active_solver : Solver<System<active>>
                      built once, reused; owns its tape; re-seeded and re-fed the
                      recording every call; holds no semantics between calls
```

Two ownership rules carry the whole design:

- **The double solver owns the recording.** The adaptive pass discovers where to place
  nodes; that placement is recorded and immutable thereafter. The active solver never
  records — it is handed the recording per call.
- **The Solver owns the AD scratch.** The active solver and its tape live on the double
  `Solver` object (not an R handle), so a C++ caller holding the solver as a plain member
  gets the reuse for free. Reuse is pure speed — it never changes a number.

## Control flow of a gradient

The primary workflow (a resident/emergent gradient) is two calls:

```
1. d.advance_adaptive({0, T})          discover the schedule, record L1/L2/L3.
                                        d is immutable hereafter.
2. compute_gradient(d, targets, schedule, functional):
     active = d.active_solver()         lift-or-reuse
     feed active the recording          (positions for a live replay; values for frozen)
     tape on; per output row:
         seed targets → active.system   (set_param / set_ic)
         active.reset()
         active.advance_fixed(schedule)  ◀── the DRIVER owns the replay
         functional(active)              ◀── a PURE REDUCTION: reads state, returns scalar(s)
     sweep adjoints → gradient
     tape off → return doubles
```

The functional never drives the solver and never carries the schedule. `recorded_steps()`
is the single source of the replay grid, so it can't go inconsistent, and the "forgot to
record" guard is one check at the single place the replay happens (`schedule.empty()`).

---

## The System contract

A System is an ODE right-hand side plus the state it integrates. To simulate, it provides
the ODE interface (`ode_size`, `set_ode_state`, `ode_rates`, `ode_state`). To be
**differentiable**, it adds:

```cpp
using value_type = S;                                  // the scalar it carries
template <class S2> using rebind = System<…, S2>;      // the double -> active mould
template <class S2> rebind<S2> rebind_from() const;    // copy config (values only) into S2
void set_param(int i, S v);                            // seed one active parameter
void set_ic(int j, S v);                               // seed one active initial-state value
```

That is the whole contract for a bare ODE. `rebind_from` copies configuration values only,
so the active system starts free of tape identity; the driver seeds the active inputs
afterwards via `set_param` / `set_ic`. Nothing forces a System to be differentiable — one
without these still simulates.

## Differentiation targets and the drivers

```cpp
compute_jacobian(solver, targets, schedule, functional);  // m x n, row i = d(out_i)/d(in)
compute_gradient(solver, targets, schedule, functional);  // the one-row case
```

**`targets`** names the inputs directly — no opaque flat-slot space:

```cpp
struct DifferentiationTargets {
  std::vector<int>    params;   // param indices to seed active
  std::vector<int>    ics;      // initial-state indices to seed active
  std::vector<double> values;   // seed values, params-then-ics
};
```

Seed only `params` for trait sensitivity, only `ics` for initial-condition sensitivity, or
both. The driver seeds each via `set_param` / `set_ic`; column *j* of the Jacobian follows
`values` in params-then-ics order.

**The functional is a pure reduction** — a callable that reads an already-replayed `Solver`
and returns the scalar(s) to differentiate. It reports its own output count:

```cpp
struct final_state {
  std::size_t codomain() const { return m; }             // how many outputs it returns
  auto operator()(auto& s) const { return s.state(); }   // reads state, returns them
};
```

Reading the count off the functional means it and the outputs it returns cannot silently
disagree, and it saves XAD an extra forward callback just to size the output (a wasted full
model replay). The record-once / row-sweep is the vendored `xad::computeJacobian`'s; odelia
supplies only the forward callback. Reverse mode is optimal here: many inputs, few outputs.

`least_squares` is one prebuilt functional (a scalar loss): it holds measured data and
scores the replayed trajectory against it. The solver stores no fit state — a custom
likelihood is just another struct with `(obs_indices, data, residual → scalar)`.

---

## Record → replay (adaptive numerics)

A bare ODE needs nothing more. A System whose rates read an **adaptively-built** field — an
interpolator, a quadrature — must freeze that construction, or the parameter-dependent
refinement branching corrupts the tape. It opts in with the `Replayable` hooks:

| Hook | Fires | Job |
|---|---|---|
| `record_stage(k)` | per RK stage, record pass | stash node positions / this stage's field value |
| `record_ode_step()` | per accepted step, record pass | commit the step's recording |
| `replay_step()` | per step, active pass | load this step's recorded slice |
| `has_recorded_field()` | query | is the L3 field cache populated? |

Two cadences, and they are the whole model:

- **Positions freeze per step** (L2). The only job is to keep the adaptive branching off the
  tape; one representative position set per accepted step does it, and it is representative
  by construction — the step controller only accepts a step whose state varies within
  tolerance.
- **Values freeze per stage** (L3). A frozen background must read back the *exact* value each
  RK stage consumed — a per-step value would be wrong, not merely coarse.

**Resident vs mutant is data presence, not a mode.** `has_recorded_field()` asks one thing:
is the L3 field cache populated?

- **empty** → the System recomputes the field on the frozen positions with the active
  scalar, so self-feedback flows (the resident/total gradient).
- **populated** → the System reads the recorded values as `double` background, off the tape,
  so the field's derivative is zero (the mutant/invasion gradient).

The variant entry chooses whether to populate the L3 cache; the System just reads what is
present. There is no `live | frozen | replaying` flag.

## The interpolator

One type carries both halves of the seam: `construct` adaptively refines a target to
tolerance (the double, record path) and `init` builds on given knots (the fixed-build,
replay path, any scalar `S`). Positions stay `double`; only the values go active. A recorded
interpolator is never re-refined.

## Off-tape edges

`SuppliedDerivative` (built on `xad::CheckpointCallback`) injects the adjoint of a value the
forward pass computed **off** the tape — a root-find, an optimiser result — carrying its
known partials into the reverse sweep, so the internal solve is never recorded.

## The R boundary

Only `double` crosses to R. R holds the double `Solver`; a gradient call builds the active
solver internally, differentiates, and returns doubles — R never holds an active type, so
the boundary can only fail loudly, never reinterpret a handle. The active solver and its
tape are cached on the double `Solver` and reused across calls, so an optimiser loop
amortizes them.

---

## Building an AD-compatible System

**A bare ODE (no adaptive sub-numerics).** Implement the ODE interface plus the four contract
members — `value_type`, `rebind` / `rebind_from`, `set_param`, `set_ic`. That's it:
`compute_gradient` / `compute_jacobian` work, doubles in and out, and a downstream package
gets `Solver_gradient` / `Solver_jacobian` for free. odelia never learns what your state
means. (`LorenzSystem` is the worked example.)

**Rates that read an adaptively-refined field.** Add the `Replayable` hooks. Record the node
**positions** your refiner chose (per accepted step) and the field **value** per RK stage;
on replay, rebuild the field on the frozen positions with the active scalar. This is
switched on by *doing reverse-mode AD*, not by any user flag — miss it and gradients are
silently wrong wherever the adaptive component bites. (`RelaxationSystem` is the worked
example.)

**A variant that holds a quantity fixed.** If you also want a cheap run that reads some
recomputable quantity as a constant (its derivative zero) — a rare mutant reading a fixed
resident field, or frozen state variables — record it per stage and have the variant entry
populate the L3 cache; `has_recorded_field()` reports it. The quantity need not be an
"environment" or an interpolator: L3 is "read a recorded `double` from a container."

odelia stays agnostic to all of it — it records "some positions" and "some values" and never
learns a node is a height or a value a field.
