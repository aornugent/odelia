# Reverse-mode gradient layer for the characteristic solver — odelia anchoring

**Status:** design / proposal. Not implemented in this PR.
**Provenance:** adapted from an external, domain-agnostic implementation spec for adding
reverse-mode parameter gradients to a method-of-characteristics transport solver via the
XAD external-function tape. The upstream spec is kept verbatim
alongside this file as [`REVERSE_GRADIENT_LAYER_SPEC.md`](./REVERSE_GRADIENT_LAYER_SPEC.md);
this document keeps that design intact and **anchors every generic construct to odelia's
real types**, then records what fits our system and the one place it does not.
**Companion docs:** [`AUTODIFF.md`](./AUTODIFF.md) (the AD surface), the shipped
census-gradient solution in plant (`docs/ad-census-gradients.md` in the `plant-dev`
superproject), and the AD PR stack (odelia#37 `reserve_state`, #40 `compute_jvp`,
#41 interpolator active-query; plant#41 the census gradient).

---

## §0 — Fit to odelia (the review)

**Verdict.** The design's *mechanism* is fully supported and already precedented in
odelia; one *implementation detail* (CB-1's exact spline transpose) assumes an
interpolator API odelia does not expose and must be realized differently. The design is
also a **different tradeoff** from the census gradient already shipped in plant — worth
landing precisely because it is the "don't change the forward model" alternative.

### What maps 1:1

| spec construct | odelia reality |
|---|---|
| XAD external-function callback with hand-written reverse | `xad::CheckpointCallback<Tape>::computeAdjoint(Tape*)` — vendored at `inst/include/XAD/`; **already wrapped** by `odelia::ode::SuppliedDerivative` (`inst/include/odelia/supplied_derivative.hpp`). CB-1/CB-2 are vector-valued siblings of it. |
| `insertCallback`, `getAndResetOutputAdjoint`, `incrementAdjoint`, output-slot creation, `getSlot` | all on `xad::Tape` (`inst/include/XAD/Tape.hpp`); ownership via `pushCallback` (tape frees it), exactly as `supplied_derivative()` does. |
| reverse driver (all θ in O(1) solves) | `odelia::ode::compute_gradient` / `compute_jacobian` (`inst/include/odelia/gradient.hpp`) on `xad::adj<double>::active_type`; tape cached on the `Solver`, row sweep delegated to `xad::computeJacobian`. |
| FD-free adjoint-identity oracle (U1.c/U2.c/V1/V7) | `odelia::ode::compute_jvp` + the `⟨Jv,u⟩=⟨v,Jᵀu⟩` dot-product (odelia#40) — no perturbation, no inner re-solve. |
| "preallocate at N_max; never resize an `AReal` container mid-recording" (P0.3) | `Solver::reserve_state(n)` (`ode_solver.hpp`): reserve the schedule-known max once, then in-place `resize()` never reallocates, so tape slots stay pinned. Verified in odelia#37 — the tape survives the growth even without it. |
| two-pass record → frozen replay (P0.2, §3.4) | L1 schedule is `Solver::set_schedule` / `replay_schedule_` + `run()`; per-stage/step recording hooks are the `Replayable` concept (`ode_interface.hpp`): `record_stage` / `record_ode_step` / `replay_step` / `has_recorded_field`. |
| segment checkpointing (P6) | `xad::Tape::getPosition` / `resetTo` / `computeAdjointsTo` and `newNestedRecording` / `xad::ScopedNestedRecording` are all declared. |
| forward-over-reverse for `g` partials | `odelia::ad::directional_derivative` / `tangent_of` / `seed` / `constant` (`directional_derivative.hpp`), type `FReal<AReal<double>>`; value-only decisions via `util::to_passive`. |

### The one real gap — CB-1's exact spline transpose (F1–F6)

The spec's CB-1 forms `c̄ = Σᵢ S̄ᵢ B(x̂ᵢ)`, `s̄ = M⁻ᵀ c̄`, then scatters through κ rows —
i.e. it needs the **collocation matrix `M`, its transpose solve, and basis rows
`B(x)`/`B′(x)`**. odelia's `basic_spline`/`basic_interpolator` expose **none of these**: it
is a natural cubic spline whose `band_matrix` is LU-solved *in place* (the factor is a
local temporary, not a member), and it is differentiable **in the knot values only**
(positions are frozen `double`). Its public reads are `operator()`, `deriv()`, and the
secant `slope(u, step, direction)`, plus the frozen-query-by-default active overload and
the opt-in `eval_with_query_derivative` (odelia#38, PR #41).

Consequences, in order of preference:

1. **Correctness does not require CB-1's hand-written transpose.** odelia already
   differentiates the field pipeline through the ordinary tape — the constant-`double`
   banded solve runs under the active scalar, carrying the knot-value derivatives — and
   the K93 census gradient already closes this way (plant#41; odelia#41 says the gather
   edge is "not a correctness dependency"). So CB-1 can be **deferred**: keep the field
   reads on the ordinary tape and ship CB-2 alone.
2. **CB-1 is the O(N·k)-vs-O(N²) performance win**, and it is exactly odelia#39's
   already-scoped "gather edge: one injected rank-k tape node via a multi-output
   `CheckpointCallback`." Building it means **adding accessors to the interpolator**
   (basis rows + a banded transpose-solve reusing the LU factor), which is odelia-side
   work and the substance of odelia#39.
3. The spec's own named fallback — an exact **pairwise one-sided sum** (O(N²),
   reconstruction-free) — sidesteps the spline entirely and is a drop-in *inside* CB-1;
   viable for validation or small N.

### The design decision this surfaces (vs what shipped)

The spec's **headline path (V4b)** keeps the forward **bit-identical** to production
(`C-VALUES`): the compression *value* stays the production stencil `Dᵢ`, and only the
*reverse* substitutes the geometric linearization `D̃ᵢ`. That leaves a bounded,
monitored bias (**A1**, the secant gap `Dᵢ − D̃ᵢ`), claimed ≤1%, shrinking under
refinement. The spec's **V4a** (test-only `--compression=geometric`) uses `D̃ᵢ` *as the
value too* and is machine-exact.

What plant shipped (`node_geometric_compression`, plant#41) **is V4a made permanent** —
`D̃ᵢ` for value and derivative, machine-exact, at the cost of moving the K93 forward
~0.2% (hence opt-in). The spec's V4b is the **opposite tradeoff**: no forward change, a
small bounded bias, purely reverse-side.

Note: an earlier plant probe of a value/derivative seam ("keep the production value,
inject a well-conditioned derivative") failed at ~90% error — but that probe kept the
*ill-conditioned spectral-slope θ-derivative* on the tape. The spec's V4b removes that
object entirely (CB-2 linearizes `D̃`, and CB-1 — or the ordinary-tape knot-value
channel — replaces the spectral-slope construction). **A properly-built V4b is untested**
and is the reason to land this: it would give correct gradients *without* touching the
published forward model.

### Relationship to the existing odelia AD stack

- odelia#37 (`reserve_state`) — the growing-dimension precondition. **Prerequisite; done.**
- odelia#40 (`compute_jvp` oracle) — the V-suite's FD-free instrument. **Done.**
- odelia#41 (interpolator active-query freeze) — the "scatter side"; makes the
  query-derivative footgun unreachable. **Done.** CB-1 is the complementary "gather side."
- odelia#39 (gather edge) — **this design's CB-1 is its concrete plan.**
- CB-1/CB-2 *bodies* (κ, `g` partials, the birth formula) are **plant-side**; odelia's
  responsibility is the generic seam: the multi-output callback primitive (generalize
  `supplied_derivative` to vector in/out) and, for CB-1, the interpolator transpose
  accessors.

### Constraint check (spec §2 hard constraints vs odelia)

- `C-VALUES` (forward bit-identical): honored by V4b; enforced by the P0.4 replay hash.
  ✓ (this is the whole point vs the shipped V4a.)
- `C-STATE` (`(x, ℓ)` state): matches plant's log-density state. ✓
- `C-FRAMEWORK` (XAD only): ✓; plus the odelia **single-`active_tape_` / `extern template`
  invariant** (ARCHITECTURE.md) — any new tape instantiation type must be added in
  `src/Tape.cpp`. CB-1/CB-2 reuse the existing `AReal<double>` tape, so no new type. ✓
- `C-COST` (O(1) solves/θ; manageable memory): ✓ via `compute_gradient` + checkpointing.
- `C-BOTH-CLASSES`: the V5 null-channel probe maps to `compute_jvp`/FD on a coupling-only
  parameter; the domain facts it relies on (`κ(z,z)=0`, order invariance, no crossings)
  are plant/K93 properties — confirm `∂ₓκ(z,z)` for A3 on the crown kernel `Q(z,H)`.

---

## §1 — Anchoring map for Part II (generic → odelia)

Use this table while reading the phase plan below; the spec's generic names are on the
left, the odelia symbol to write is on the right.

| spec name | odelia symbol / file |
|---|---|
| `Real` templated scalar | `xad::adj<double>::active_type` (pass 2); `double` (pass 1) |
| `passive(v)` | `xad::value(v)` for one layer; `odelia::util::to_passive(v)` to strip *all* AD layers (nesting-safe) |
| slot for an active | `active.getSlot()`; `Tape::registerInput(active)` to make a leaf |
| callback base | `xad::CheckpointCallback<Tape>` (subclass; override `computeAdjoint`) |
| insert + own a callback | `tape.pushCallback(cb); tape.insertCallback(cb);` |
| `getAndResetOutputAdjoint`, `incrementAdjoint` | same names on `xad::Tape` |
| record schedule | `Solver::set_schedule` / `replay_schedule_`; `Replayable` hooks |
| preallocate at N_max | `Solver::reserve_state(N_max)` |
| reverse sweep / harvest | `compute_gradient` / `compute_jacobian`; `xad::derivative(θ_slot)` |
| JVP oracle | `compute_jvp` / `compute_directional_derivative` |
| `g` partials `g_x,g_S,g_θ` (P0.1, CB-2) | closed form for K93; `directional_derivative` for the mass cascade; `supplied_derivative` (envelope theorem) for TF24's leaf-optimiser `g` |
| checkpoint position/rewind (P6) | `Tape::getPosition` / `resetTo` / `computeAdjointsTo`; `ScopedNestedRecording` |
| spline reads `S(x)`, `S′(x)` | `interpolator.eval(u)` / `deriv(u)` / `slope(u,step,dir)`; **no `M`, `M⁻ᵀ`, or basis-row accessor exists — see §0 gap** |

---

## §2 — The design (odelia-anchored summary)

Two callback seams per RK stage; everything else is plain active arithmetic on the
existing tape. Both are vector-valued generalizations of `odelia::ode::SuppliedDerivative`.

- **CB-1 `CouplingFieldOp`** — forward: the unmodified production field pipeline on
  passive values, outputs `Sᵢ = S(xᵢ)` (and `S_b` on birth steps) set exactly and made
  tape leaves via `registerInput`. Reverse: the transpose of the pipeline (gather reader
  adjoints → coefficient space → knot-sample cotangents → scatter through κ deposits).
  **odelia note:** the transpose needs the interpolator collocation API odelia does not
  expose (§0 gap). Correctness alternative: skip CB-1 and let the ordinary tape carry the
  field reads' knot-value derivatives (the frozen-query default of odelia#41); CB-1 then
  becomes the odelia#39 gather-edge performance optimization.
- **CB-2 `CompressionOp`** — forward: the production compression *value* `Dᵢ` (bit-exact,
  keeps `C-VALUES`). Reverse: the linearization of the **geometric** secant
  `D̃ᵢ = [g(x_{i+1},S_{i+1}) − g(x_{i−1},S_{i−1})]/(x_{i+1} − x_{i−1})`. This restores, in
  the linearization of every `mᵢ = e^{ℓᵢ}Δxᵢ`, the compression cancellation the continuum
  guarantees. `g` partials via the anchoring-map row above. **This is the reverse-only
  form of what plant shipped as a forward change** (`node_geometric_compression`).

**Commitment (kept true by structure):** the field pipeline and compression stencil are
reachable only through the two callbacks — they stay `double`-only (not templated), so no
active variable can enter them except through a hand-written reverse. This mirrors
odelia's existing rule that AD is glue around XAD facilities (`AGENTS.md`).

The returned gradient is the exact discrete adjoint of the **geometric-compression model**
linearized along the recorded production trajectory; §0 explains why that is the right
target (moments are transport-free; the geometric pairing preserves the cancellation).

---

## §3 — Approximation ledger (unchanged from the spec; diagnostics anchored)

| id | approximation | odelia diagnostic |
|---|---|---|
| **A1** | value = production stencil `Dᵢ`, linearization = geometric `D̃ᵢ` | per-stage `max`/`rms` of `|Dᵢ − D̃ᵢ|`, running `Γ = Σ h·maxᵢ|Dᵢ−D̃ᵢ|`; V4a (machine-exact) vs V4b (≤1%) isolates it. In plant's shipped V4a this term is **zero** (value is also `D̃`). |
| **A2** | frozen-schedule gradient; schedule θ-sensitivity dropped | gradient spread across 2–3 `advance_adaptive` recording tolerances (V6). |
| **A3** | deposit-derivative kink at knot crossings, only if `∂ₓκ(z,z)≠0` | near-knot counter; check the crown kernel's `∂ₓκ` on the diagonal (plant). |

No self-exclusion / cavity / masking anywhere — the pathological self-force lived only in
the θ-derivative of the spectral-slope construction, which CB-2 removes from the
derivative path.

---

## §4 — Phase plan, anchored (P0–P7)

The upstream spec's phases are sound; the only edits are odelia symbols and the CB-1 gap.

- **P0 Prerequisites.** P0.1 scalar templating already exists in plant (the rate path is
  `value_type`-templated; the field pipeline/stencil stay `double` behind the callbacks).
  P0.2 schedule = `replay_schedule_`; enforce "no active-value branch in pass 2" as the
  spec says. **P0.3 preallocation = `Solver::reserve_state(N_max)`** (odelia#37; already
  proven). P0.4 standing assertions (order invariance; bitwise replay hash) — add to the
  `Replayable` hooks. P0.5 θ registration = `DifferentiationTargets` + the tape inputs
  `compute_gradient` already seeds.
  **Gate P0:** pass 2 at `AReal` with no callbacks runs the frozen schedule green.
- **P1 CB-1 `CouplingFieldOp`.** Build as a vector `SuppliedDerivative`. *Gated on the §0
  interpolator decision* — either add basis-row + banded-transpose accessors to
  `basic_spline`/`basic_interpolator` (odelia#39), or ship correctness through the
  ordinary tape and treat CB-1 as the later perf optimization. Unit tests U1.a–d;
  **U1.c adjoint identity uses `compute_jvp`** (odelia#40).
- **P2 CB-2 `CompressionOp`.** No interpolator dependency; buildable now. `g` partials per
  the anchoring map. Diagnostic D1 (`Γ`). U2.a–d; U2.c via `compute_jvp`.
- **P3 Birth map.** The production birth formula in taped `Real`, consuming CB-1's `S_b`;
  plant-side. Causality probe as specified.
- **P4 Driver.** The pass-2 loop is plant's `Species/Patch::compute_rates` under the
  active scalar; the sweep is `compute_gradient` per functional. Callback lifetime is the
  tape's (`pushCallback`).
- **P5 Verification.** V1/V7 adjoint identities via `compute_jvp`; V2/V3 the replay hash /
  order invariance; **V4a** = plant's shipped `node_geometric_compression=TRUE` path
  (machine-exact, already achieved); **V4b** = production value + CB-2 reverse (the new,
  untested ≤1% path — the deliverable to validate); V5 null-channel; V6 tolerance study;
  V7 the `--coupling=taped` A/B (ordinary tape vs CB-1 transpose) — only meaningful once
  CB-1 exists.
- **P6 Memory/checkpointing.** Measure `tape.getMemory()`; if over budget, segment at
  birth joints using `getPosition`/`resetTo`/`computeAdjointsTo` + nested recording,
  respecting the single-`active_tape_` invariant.
- **P7 Rollout.** Flags mirror plant's existing `node_geometric_compression` (V4a) plus a
  new reverse-only V4b mode and the `--coupling=callback|taped` and `--checkpoint`
  switches.

---

## §5 — XAD API checklist (spec Appendix B), resolved against odelia

Every item the spec says to "verify against your XAD version" is answered here, so no
verification step is needed before P1:

1. Callback base/virtual: `xad::CheckpointCallback<Tape>` / `computeAdjoint(Tape*)`. ✓
2. Slot for an active / input registration: `active.getSlot()`, `Tape::registerInput`. ✓
3. Callback outputs (fresh actives, value set, tape-visible): `registerInput(y)` on a
   fresh active whose value is set (the `supplied_derivative` idiom). ✓
4. `getAndResetOutputAdjoint(slot)` with reset-on-fetch: present; used by
   `SuppliedDerivative`. ✓
5. `incrementAdjoint(slot, double)`: present. ✓
6. LIFO callback order during the sweep (CB-2 → CB-1 composition): callbacks run in
   reverse insertion order — **still write the two-callback toy test** the spec asks for,
   as a standing guard. ✓ (mechanism present)
7. Ownership/lifetime: `pushCallback` hands ownership to the tape (freed with it). ✓
8. Position/rewind for nested re-recording (P6): `getPosition` / `resetTo` /
   `computeAdjointsTo`, `newNestedRecording` / `ScopedNestedRecording`. ✓
9. Forward-mode type for the JVP tests: `xad::fwd<double>::active_type` via `compute_jvp`;
   forward-over-reverse via `odelia::ad::directional_derivative`. ✓
10. `xad::value()` inside a recording (must not record): used throughout; `util::to_passive`
    for full multi-layer stripping. ✓

---

## §6 — What this PR is and is not

**Is:** the odelia-anchored design for the reverse-mode gradient layer — the concrete plan
for odelia#39's gather edge (CB-1) and a compression callback (CB-2), a fit review, and
the corrected account of the one interpolator gap. A starting point for implementation.

**Is not:** an implementation. CB-1/CB-2 bodies are plant-side and a multi-week effort;
the interpolator transpose accessors (if CB-1 is built) are the odelia#39 work item. No
code changes ship here beyond this document.

**Recommended first step** (smallest, correctness-relevant, no interpolator dependency):
implement **CB-2 alone** as a reverse-only alternative to plant's `node_geometric_compression`,
and run **V4b** to measure whether the forward-pristine path holds ≤1% at production
resolution. That single measurement decides whether the "gradients without a forward
change" path is viable — the question the shipped V4a deliberately sidesteps by changing
the forward.
