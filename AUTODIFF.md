# Automatic differentiation in odelia: exact gradients of an ODE run

**Scope:** `traitecoevo/odelia` — the reverse-mode automatic-differentiation API. A
new developer with only this code should come away understanding what a gradient is made
of, who owns what, the two axes every workflow is a point on, the contract a System
implements to be differentiable, and the one way to get an adaptive-component gradient
wrong. Self-contained.

**Companion:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the XAD `Tape` runtime is
compiled once and linked across the DLL boundary. Read it before touching the build;
this document is about the API, not the link.

---

## The one-sentence design

odelia differentiates **any reduction of an ODE run** — on one scalar-templated
`Solver`, with reverse-mode XAD compiled once — and only `double` ever crosses back to R.

Reverse mode (a *tape* records each operation, then sweeps backward to get every input's
derivative from one output) is the right engine here because there are many inputs
(parameters, initial conditions) and few outputs: one backward sweep yields the whole
gradient. XAD is the vendored AD library; odelia compiles its `Tape` once and is glue
around it, never a second AD engine.

Everything else follows: the System is templated on its scalar so the same code simulates
(`double`) and differentiates (`active`); the thing differentiated is a caller's
functional, not a built-in loss; adaptive numerics are recorded once and replayed on fixed
nodes so the tape carries no adaptive branching; and the active machinery is born, used,
and destroyed inside one C++ call.

---

## The mental model: two orthogonal axes

Most of the apparent complexity dissolves once you see that a gradient workflow is a choice
on **two independent axes**:

- **Replay** — which adaptive constructions the *system* must record on the double pass and
  replay *fixed* on the active pass. A property of the system, not the question. A plain
  ODE needs only the step schedule (L1). A system whose rates read an adaptively-built
  background also records that background's node positions (L2), and optionally its values
  (L3) for a variant that holds the background fixed.
- **Functional** — what scalar you reduce the replayed run to: an emergent summary of the
  state, or a likelihood over measured data. Orthogonal to replay.

Three representative points:

| Application | What you differentiate | Replay | Functional |
|---|---|---|---|
| **Sensitivity** | d(output)/d(param or IC) of a plain ODE | L1 | any reduction |
| **Adaptive-component gradient** | the same, when a rate reads an adaptively-built background | L1·L2 (+L3 to freeze it) | any reduction |
| **Calibration / inference** | fit params to measured data | whatever the system needs | a likelihood |

The first two differ only on the replay axis (how complex the system is); the third differs
only on the functional axis. Calibration is not a different replay — it is the system's own
replay with a likelihood functional on top.

---

## Ownership

R holds one object — the ordinary (double) `Solver`. Everything active hangs off it and
never surfaces:

```
R  ── holds ──▶  d : Solver<System<double>>          the double solver
                 │
                 ├─ System<double>                    immutable after the adaptive pass
                 ├─ recording                          produced once, read per call:
                 │    ├─ L1  times() / step_sizes()    reached times and step sizes (Solver state)
                 │    ├─ L2  positions_history[step]   adaptive node positions   (System state)
                 │    └─ L3  values_history[step][stage] recorded background values (System state)
                 └─ active_solver : Solver<System<active>>
                      built once, reused; owns its tape; re-seeded and re-fed the
                      recording every call; holds no semantics between calls
```

Three ownership rules carry the whole design:

- **The Solver owns the schedule (L1); the System owns its background (L2/L3).** The stepper
  steps any system either adaptively (discovering node positions) or on a fixed grid
  (replaying them). What a rate reads is the System's own business; the stepper only signals
  cadence through the hooks.
- **The double solver owns the recording, and it is immutable after the adaptive pass.** Its
  only remaining job is to feed replays. The recording is keyed to that run's ICs and
  params; change those and you must re-record. The active solver never records — it is
  handed the recording, per call.
- **The Solver owns the AD scratch.** The active solver and its tape live on the double
  `Solver` object (not an R handle), so a C++ caller holding the solver as a plain member
  gets the reuse for free. Reuse is pure speed — it never changes a number. The recording,
  read per call, is what carries semantics.

## Control flow of a gradient

```
1. d.advance_adaptive({0, T})          discover the schedule, record L1/L2/L3.
                                        d is immutable hereafter.
2. compute_gradient(d, targets, schedule, functional):
     active = d.active_solver()         lift-or-reuse
     feed active the recording
     tape on; per output row:
         seed targets → active.system   (ad_parameters / ad_initial_state)
         active.reset()
         active.advance_fixed(schedule)  ◀── the DRIVER owns the replay
         functional(active)              ◀── a PURE REDUCTION: reads state, returns scalar(s)
     sweep adjoints → gradient
     tape off → return doubles
```

The functional never drives the solver and never carries the schedule. `recorded_steps()`
is the single source of the replay grid, so it can't go inconsistent, and the "forgot to
record" guard is one check at the single place the replay happens (`schedule.empty()`).

### What a recording holds, and what a replay may drive from

An adaptive pass records `(t, h, y)` at every accepted step: the time it reached
(`times()`), the size of the step that reached it (`step_sizes()`), and the state there.
Both `t` and `h` are recorded because neither determines the other in floating point.
`fl(fl(t + h) - t) != h` -- the addition rounds away at `ulp(t)` and the subtraction cannot
recover the discarded low bits of `h`. So a replay that recovers each step by differencing
recorded times integrates a slightly different trajectory than the one recorded, silently.
`advance_fixed_steps` exists for the replay that must be faithful to the recorded run: it
steps by the recorded `h` and reproduces that run bitwise.

The schedule the gradient driver replays is a caller-supplied time grid, and
`advance_fixed` drives it by times. That is correct for that operation -- the grid *is* the
specification, and there is no adaptive run for it to be faithful to.

**Two operations sit behind one `set_schedule()` / `run()`.** One is *replay this recorded
adaptive trajectory*, whose grid is the recorded step sizes; the other is *differentiate a
solve over this time grid*, whose grid is the caller's times. They are not the same
schedule and they are not driven the same way, and only the caller knows which it means.
Two explicit entry points are owed here, one per operation.

---

## The System contract

A System is an ODE right-hand side plus the state it integrates. To simulate, it provides
the ODE interface (`ode_size`, `set_ode_state`, `ode_rates`, `ode_state`).

A System built from a **range of elements** hands that range to the helpers in
`ode_interface.hpp`, which walk it and thread one iterator through. Those helpers require
`OdeElement`: an element names its own `value_type` and moves its state through an iterator
over that type, not over `double`. The requirement is on the iterator rather than on the
members, because a missing member reports itself while a wrong iterator type reports a page
of instantiation errors far from the cause.

The aux family is five, not four: `aux_size`, `ode_aux` and `set_ode_aux` alongside
`ode_size`, `ode_state`, `ode_rates` and `set_ode_state`. `set_ode_aux` is an ordinary
member rather than an opt-in, so the solver can assert of it what it asserts of the others —
that the iterator advanced by `aux_size()`. It exists so a System can hand back a quantity
that is expensive to recompute and cheap to carry, per step or per stage, with a width the
solver checks. A System that publishes nothing to aux pays nothing.

To be **differentiable**, a System adds:

```cpp
using value_type = S;                                  // the scalar it carries
template <class S2> using rebind = System<…, S2>;      // the double -> active mould
template <class S2> rebind<S2> rebind_from() const;    // copy config (values only) into S2
std::vector<S*> ad_parameters();                       // handles to the active parameters
std::vector<S*> ad_initial_state();                    // handles to the active initial state
```

That is the whole contract for a bare ODE. `rebind_from` copies configuration values only,
so the active system starts free of tape identity; the driver then seeds a chosen subset of
inputs by writing through the handles `ad_parameters()` / `ad_initial_state()` return (in a
fixed order, so a 30-parameter System is one `return {&a, &b, …}` rather than a switch).
Nothing forces a System to be differentiable — one without these still simulates.
(`LorenzSystem` is the worked example.)

### The twin a stage recording is taken on: `rebind_from`, and `seat_from`

A reverse sweep needs, at each stage, the transpose of the rate evaluation: the adjoints of
`dydt` taken back to the adjoints of `y`, one row per seed. `Step::step_adjoint_batched` builds
that itself. It rebuilds the six stage states in `value_type`, then for each of them records
`derivs()` once on the System lifted to the adjoint scalar and sweeps that one recording per
seed, so the caller's sweep stays in `value_type`. What is recorded is the call the forward
pass makes, so the transpose cannot drift from the rates it transposes, and a System that
restores a recorded field restores it there as well. The state and the twin's
`ad_parameters()` are recorded together, so the stage carries parameter rows as well as state
rows. That needs nothing from the System but `rebind_from`, and a `static_assert` on
`Rebindable` says so where it is missing.

`twin` is the System at the adjoint scalar, owned by the caller and handed to every stage. Each
recording clears the tape, so a twin arriving with the previous recording's slots writes this
recording's operations onto numbers already handed out and the sweep comes back wrong with
nothing raised — one seed's rows exact and another's not. It is therefore re-seated before
every recording, and there are six of those per step. A System that can be re-seated in place
declares

```cpp
template <class S2> void seat_from(const System<…, S2>& src);
```

which writes the values `rebind_from` copies into a twin that already exists. Satisfying
`SeatsFrom` (in `ode_jacobian.hpp`) switches `seat_twin` onto it with `if constexpr` — a
compile-time choice; a System without it is rebound instead. The two leave the twin holding the
same thing, so the choice is a cost one: for a System whose copy allocates per element, a
rebind per recording costs what the recording costs.

`Step::stage_sweeps` counts the stage transposes taken, seeds counted separately, and
`Solver::stage_sweeps()` reads it through. A row that enters once per stage is multiplied by
that count, and a row correct per evaluation and wrong in its multiplier is a different failure
from a wrong row: no gradient check can see it, because a tangent and a sweep apply the same
multiplier.

**The adjoint is RKCK only.** `method = "rodas"` has no reverse counterpart and
`Solver::step_adjoint_batched` stops rather than stepping — a Rosenbrock stage is a linear
solve, and transposing it is a different derivation, not the same one with a sign moved.

`Solver::solve_adjoint(states, lambda)` drives the sweep over a whole run: it walks
`recorded_steps()` last to first, so on return `lambda` is the adjoint of `states[0]`. Only
accepted steps are recorded and a rejected step never enters the solution, so the recorded list
is the whole of what the sweep visits. It stops if there is no recording, which is the
"forgot to record" guard rather than a sweep over one step.

**A System that changes width mid-run sweeps one segment per width.**
`solve_adjoint(states, lambda, k_first, k_last)` restricts the sweep to steps `k_last` down to
`k_first + 1`, so on return `lambda` is the adjoint of `states[k_first]`; the two-argument form
is that call over the whole recording. Every state a segment visits has to be the width the
System holds, so a caller whose System widens or narrows between steps runs a segment, changes
the System, and runs the next — and it changes the System itself, because the solver has no way
to know what a width change *means*. A caller that adds a component to a compound state knows
which rows are new and which carried over; a width alone under-determines that, so a rule like
"drop the last row" is wrong for any state that is not laid out in the order the additions
happened. **The narrowing and the between-segment boundary condition belong to the System's
owner, and only the segment range belongs here.**

The stage buffers are sized at construction, so `Solver::step_adjoint_batched` resizes them from
the adjoints it is handed rather than from the width the forward pass left. A sweep is the end of
the solver's forward state in any case: every stage rebuild overwrites it.

Two properties hold and both are tested. Every interior split of one recording sweeps
**bit-identically** to the whole sweep — the segment boundary is a place the sweep is
interrupted, not a place it is approximated. And a range that is not a range of recorded steps
stops rather than sweeping a truncated one.

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

Seed only `params`, only `ics`, or both. The driver writes each through the handles
`ad_parameters()` / `ad_initial_state()` return.

> **Column order is a contract.** Jacobian column *j* is `d(output)/d(input_j)` for the
> *j*-th seeded input, in `values` order (params then ics). A caller that resolves names →
> indices must seed in the same order it reads columns, or the columns transpose silently.

## Functionals: what you differentiate

A functional is **any callable** that reads an already-replayed `Solver` and returns the
scalar(s) to differentiate, plus a `codomain()` reporting how many outputs it returns. It
does not drive the solver.

The whole final state — `m = ode_size` outputs:

```cpp
struct final_state {
  std::size_t codomain() const { return m; }
  template <class Solver>
  std::vector<typename Solver::value_type> operator()(Solver& s) const {
    return s.state();
  }
};
```

A scalar summary — the summed final state:

```cpp
struct sum_final_state {
  std::size_t codomain() const { return 1; }
  template <class Solver>
  typename Solver::value_type operator()(Solver& s) const {
    typename Solver::value_type total(0);
    for (auto const& x : s.state()) total += x;
    return total;
  }
};
```

Anything you can compute from the replayed state is a valid functional: one component, a
weighted sum, a nonlinear metric, a residual against data. You write a struct with
`codomain()` and `operator()(solver)`; odelia differentiates whatever it returns and never
learns what the scalars mean. `least_squares` is just a prebuilt one — a scalar loss that
holds measured data and scores the trajectory against it. Reading the count off the
functional (rather than passing it) means the count and the outputs cannot silently
disagree, and it saves XAD an extra forward callback — a full wasted replay — just to size
the output. The record-once / row-sweep is the vendored `xad::computeJacobian`'s; odelia
supplies only the forward callback.

---

## Record → replay: any adaptive component

A bare ODE needs nothing more than the above. But if a rate reads a background built by an
**adaptive** construction — one that makes parameter-dependent decisions about where to
place nodes — differentiating through those decisions corrupts the tape. On a gradient pass
the adaptive run has already happened, so the placement is already known: record it once,
replay pinned to it.

The **interpolator** is the running example. On the double pass it refines its knots
adaptively; on the active pass it rebuilds on the recorded knots with active values. But
nothing here is interpolator-specific: a quadrature's subdivision, or any adaptive refiner,
records through the identical hooks. The System records whatever *positions* its adaptive
machinery chose; odelia never learns a node is a knot. A System opts in with the
`Replayable` hooks:

| Hook | Fires | Job |
|---|---|---|
| `record_stage(k)` | per RK stage, record pass | stash node positions / this stage's background value |
| `record_ode_step()` | per accepted step, record pass | commit the step's recording |
| `replay_step()` | per step, active pass | load this step's recorded slice |
| `has_recorded_field()` | query | is the L3 background cache populated? |

Two cadences, and they are the whole model:

- **Positions are recorded per step** (L2). The only job is to keep the adaptive branching
  off the tape; one representative position set per accepted step does it, and it is
  representative by construction — the step controller only accepts a step whose state (hence
  the background's structure) varies within tolerance across it.
- **Values are recorded per stage** (L3). A background held fixed must read back the *exact*
  value each RK stage consumed — a per-step value would be wrong, not merely coarse.

**Whether a background's derivative flows is a data question.** `has_recorded_field()` asks
one thing: are recorded background values present (the L3 cache)?

- **empty** → the System recomputes the background on the recorded positions with the active
  scalar, so its feedback into the rates is differentiated (the derivative flows through it).
- **populated** → the System reads the recorded values as `double`, off the tape, so that
  background's derivative is zero by construction — you have held it fixed.

The caller chooses per replay whether to populate the cache. Reach for the hold-fixed
(populated) variant when a background is a shared or coupling quantity you deliberately want
to hold constant — its contribution to the derivative is then zero, not merely small.

> **The one way to get it wrong.** If you want a background's feedback in the gradient, you
> must leave L3 *empty* so it is recomputed with the active scalar on the recorded positions.
> Populating it — or otherwise reading the recorded values — drops that background's
> contribution to zero *silently*: the run succeeds and returns a plausible number that is
> missing the feedback cross term. The positions are fixed so the tape stays clean; the
> values are recomputed so the gradient flows through *what the nodes hold*, never through
> *where they sit*.

---

## Injecting a known derivative

**When you need it.** Sometimes the forward pass computes a value the tape never recorded —
an iterative solver, a root-find, an optimizer result. Differentiating *through* the
iteration is wasteful (it tapes every iterate) or impossible (the iteration count is
data-dependent), but you know the result's derivative analytically — typically from the
implicit function theorem. `SuppliedDerivative` lets you register the off-tape value as a
fresh input and hand the reverse sweep its known partials, so the internal solve is never
recorded.

**Example.** A root-find gives `x` solving `x = cos(x) + a`, and you differentiate
`g = x²` w.r.t. `a`. Nothing about the Newton loop is on the tape. By the implicit function
theorem, `dx/da = 1 / (1 + sin(x))`. Register `x` as a leaf carrying that partial; then
`g = x*x` records normally and the reverse sweep returns `dg/da = 2x · dx/da`. Built on
`xad::CheckpointCallback`, it is a free function called from *within* the forward pass —
where the off-tape value exists — not a pre-declared target. odelia sees only inputs, an
output value, and partials.

---

## The R boundary

Only `double` crosses to R. R holds the double `Solver`; a gradient call builds the active
solver internally, differentiates, and returns doubles — R never holds an active type, so
the boundary can only fail loudly, never reinterpret a handle as the wrong scalar. The
active solver and its tape are cached on the double `Solver` and reused across calls, so an
optimiser loop (asking for value and gradient each iteration) amortizes them.

---

## Building an AD-compatible System

**A bare ODE (no adaptive sub-numerics).** Implement the ODE interface plus the four
contract members — `value_type`, `rebind` / `rebind_from`, `ad_parameters`,
`ad_initial_state`. That's it: `compute_gradient` / `compute_jacobian` work, doubles in
and out, and a downstream package gets `Solver_gradient` / `Solver_jacobian` for free.

The complete member set:

```cpp
template <class S = double>
class MySystem {
public:
  using value_type = S;

  // Simulate: the ODE interface.
  std::size_t ode_size() const;                          // number of state variables
  double      ode_time() const;                          // current time
  template <class It> It set_ode_state(It y, double t);  // load state at time t; recompute rates
  template <class It> It ode_state(It out) const;        // write the current state
  template <class It> It ode_rates(It out) const;        // write the current dy/dt
  void reset();                                          // return to the initial state

  // Differentiate: copy onto another scalar, and expose the inputs to seed active.
  template <class S2> using rebind = MySystem<S2>;
  template <class S2> rebind<S2> rebind_from() const;    // xad::value the config into an S2 copy
  std::vector<S*> ad_parameters();                       // addresses of the active parameters
  std::vector<S*> ad_initial_state();                    // addresses of the active initial state
};
```

`LorenzSystem` is the worked example to copy from.

**Rates that read an adaptively-refined background.** Add the `Replayable` hooks. Record the
node **positions** your refiner chose (per accepted step) and the background **value** per
RK stage; on replay, rebuild on the recorded positions with the active scalar. This is
switched on by *doing reverse-mode AD*, not by any user flag — miss it and gradients are
silently wrong wherever the adaptive component bites.

```cpp
void record_stage(int stage);     // per RK stage, record pass: stash a value
void record_ode_step();           // per accepted step, record pass: commit the step
void replay_step();               // per step, replay pass: load the recorded slice
bool has_recorded_field() const;  // are recorded values present to reuse?
```

`CanopySystem` is the worked example.

**A variant that holds a background fixed.** If you also want a cheap run that reads some
recomputable quantity as a constant (its derivative zero) — a coupling field, held-fixed
state variables — record it per stage and have the variant entry populate the L3 cache;
`has_recorded_field()` reports it. L3 is "read a recorded `double` from a container"; the
quantity need not be an interpolated field.

odelia stays agnostic to all of it — it records "some positions" and "some values" and never
learns a node is a height or a value a field.
