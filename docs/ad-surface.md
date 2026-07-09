# odelia's AD surface: the gradient API and the System contract

**Scope:** `traitecoevo/odelia` — what odelia gives a downstream package for exact
reverse-mode gradients, and the contract a System implements to get them. The *why*
behind the shape lives in the design hub; this is the map.

**Companions:** [`ARCHITECTURE.md`](../ARCHITECTURE.md) (how the XAD `Tape` runtime is
compiled once and linked across the DLL boundary — read it before touching the build);
[`plant-dev/docs/`](https://github.com/aornugent/plant-dev/tree/main/docs) — the design:
`ad-infrastructure-design.md` (the plant emergent-gradient design this serves),
`ad-record-replay.md` (the adaptive-numerics primitive), `ad-r-interface.md` (the R/C++
boundary).

---

## The one-sentence design

odelia differentiates **any reduction of an ODE run** — on one scalar-templated
`Solver`, with reverse-mode XAD compiled once — and only `double` ever crosses back to R.

Everything below is a consequence: the System is templated on its scalar so the same
code simulates (`double`) and differentiates (`active`); the thing being differentiated
is a caller's functional, not a built-in loss; adaptive numerics are recorded once and
replayed fixed so the tape stays clean; and the active machinery is born, used, and
destroyed inside one C++ call.

---

## 1. The System contract

A System is an ODE right-hand side plus the state it integrates. To simulate, it
provides the ODE interface (`ode_size`, `set_ode_state`, `ode_rates`, `ode_state`).
To be **differentiable**, it adds four things:

```cpp
using value_type = S;                                  // the scalar it carries
template <class S2> using rebind = System<…, S2>;      // the double -> active mould
template <class S2> rebind<S2> rebind_from() const;    // copy config (values only) into S2
void scatter(Iterator values, const std::vector<int>& slots);  // route active inputs to fields
size_t n_params() const;                               // how many leaves are parameters
```

That is the whole contract — implement it and `compute_gradient` / `compute_jacobian`
work, doubles in and out. Nothing forces a System to be differentiable: a System without
these still simulates, and the gradient path is simply unavailable to it.

## 2. A functional is a pure reduction

The thing differentiated is a **functional**: a callable that reads an already-replayed
`Solver` and returns the scalar(s) to differentiate. It does not drive the solver and
carries no schedule — the driver replays, the functional reduces.

```cpp
struct final_state { auto operator()(auto& s) const { return s.state(); } };
```

odelia never learns what the scalar means — an emergent summary of the state, or a
calibration loss. `least_squares` is one prebuilt instance: it holds measured data and
scores the replayed trajectory against it; the solver stores no fit state.

## 3. The reverse-mode drivers

```cpp
compute_jacobian(solver, targets, schedule, functional, codomain);  // m x n, row i = d(out_i)/d(in)
compute_gradient(solver, targets, schedule, functional);            // the one-row case
```

- **`targets`** (`DifferentiationTargets`) are opaque `(slot, value)` leaves; the System
  routes each to a field via `scatter`, so odelia never learns whether a leaf is a trait,
  an initial density, or a flux. Column *j* of the Jacobian is `(slots[j], values[j])`.
- **`schedule`** is the recorded step grid the driver replays (`advance_fixed`); owning
  the replay here keeps the grid from drifting from the recording.
- **`codomain`** is the number of outputs *m* — pass it, or XAD runs the forward callback
  an extra time just to size the output.

The record-once / row-sweep is the vendored `xad::computeJacobian`'s; odelia supplies
only the forward callback (scatter → reset → replay → reduce). Reverse mode is optimal
here: many inputs, few outputs.

## 4. Record → replay (adaptive numerics)

A bare ODE needs nothing more. A System whose rates read an **adaptively-built** field
(an interpolator, a quadrature) must freeze that construction so the tape carries no
adaptive branching. It opts in with the `Replayable` hooks:

| Hook | Fires | Job |
|---|---|---|
| `record_stage(k)` | per RK stage, record pass | stash node positions / field value |
| `record_ode_step()` | per accepted step, record pass | commit the step's recording |
| `replay_step()` | per step, active pass | load this step's recorded slice |
| `has_recorded_field()` | query | is the L3 field cache populated? |

The double pass records **where** the adaptive scheme placed its nodes; the active pass
replays **pinned** to them, values active. `has_recorded_field()` is a data query, not a
mode: an empty L3 cache means recompute the field live (self-feedback flows); a populated
one means read it as `double` background (its derivative is zero). Full treatment:
`ad-record-replay.md`.

## 5. The interpolator

One type carries both halves: `construct` adaptively refines a target to tolerance (the
double/record path) and `init` builds on given knots (the fixed-build/replay path, any
scalar `S`). Positions stay `double`; only the values go active.

## 6. Off-tape edges

`SuppliedDerivative` (built on `xad::CheckpointCallback`) injects the adjoint of a value
the forward pass computed **off** the tape — a root-find, an optimiser result — carrying
its known partials into the reverse sweep, so the internal solve is never recorded.

## 7. The R boundary

Only `double` crosses to R. R holds the ordinary (double) `Solver`; a gradient call builds
the active solver internally, differentiates, and returns doubles — R never holds an active
type, so the boundary can only fail loudly, never reinterpret a handle. The active solver
(and its tape) is cached on the double `Solver` object and reused across calls, so an
optimiser loop amortizes it; reuse is pure speed and never changes a number. Details:
`ad-r-interface.md`.
