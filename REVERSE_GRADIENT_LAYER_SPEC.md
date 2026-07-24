# Spec: Reverse-Mode Gradient Layer for the Characteristic Solver (XAD)

> **odelia note.** This is the upstream, domain-agnostic spec, kept verbatim. For how
> each construct maps to odelia's real types (and the one place it does not — the
> interpolator collocation transpose), read [`REVERSE_GRADIENT_LAYER.md`](./REVERSE_GRADIENT_LAYER.md)
> first; it anchors this plan and reviews its fit to our system.

**Status:** implementation-ready.
**Audience:** a developer comfortable with C++, templates, and basic numerics. No prior context on this problem is assumed; Part I is the required background.
**Scope:** add reverse-mode parameter gradients to the existing characteristic solver using the XAD operator-overloading tape. The forward model, its state variables, and its numerics are **not** changed.
**Estimated effort:** 2–3 weeks including the verification suite.

---

# Part I — Background

## 1. The system being differentiated

The production solver integrates a transport (conservation) law over a scalar coordinate `x` by the method of characteristics, with a nonlocal coupling field and a boundary influx.

**Continuum model.**

```
∂ₜ n(x,t) + ∂ₓ[ g(x, S; θ) · n ] = − r(x, S; θ) · n ,        x > x_b
S(z,t) = exp( A(z,t) )
A(z,t) = ∫_{x ≥ z} κ(z, x; θ) · n(x,t) dx        (one-sided cumulative aggregate)
```

- `n(x,t)`: a density over `x`. `θ`: the parameter vector, `|θ| ≈ 5–20`.
- `g`: velocity (advects mass upward in `x`); `r`: loss rate. Both are smooth closed-form functions of the local coordinate, the field value `S`, and `θ`.
- `κ`: smooth, one-signed kernel, defined for `x ≥ z`, with **κ(z, x) → 0 as x → z⁺** (it vanishes on the diagonal). Only mass *above* `z` contributes to `A(z)`.

**Discretization (the production method, unchanged by this work).**

- The density is represented by `N(t)` **characteristics** with positions `x₁ < x₂ < … < x_N` (strictly increasing) and per-characteristic **log-densities** `ℓᵢ`.
- Rates, per characteristic:

```
dxᵢ/dt = g(xᵢ, Sᵢ; θ)                          Sᵢ = S(xᵢ), the field value read
dℓᵢ/dt = − ( Dᵢ + r(xᵢ, Sᵢ; θ) )               Dᵢ = the "compression": the spatial
                                               derivative ∂ₓ[g(x, S(x))] at xᵢ,
                                               evaluated by a finite-difference
                                               secant of the reconstructed field
                                               (a stability-mandated stencil)
```

- **Field pipeline, rebuilt every RK stage:**
  1. Quadrature weights `wⱼ = e^{ℓⱼ} · Δxⱼ`, where `Δxⱼ` is the production quadrature spacing (e.g. `(x_{j+1} − x_{j−1})/2` interior; production's own endpoint rule at the ends).
  2. Knot aggregates at `k ≈ 17` **fixed** knot positions `z₁ … z_k`:
     `A_m = Σ_{j : xⱼ ≥ z_m} κ(z_m, xⱼ; θ) · wⱼ`.
  3. Knot samples `s_m = exp(A_m)`.
  4. Spline coefficients `c = M⁻¹ s`, where `M` is the fixed `k×k` spline collocation matrix (knots fixed ⇒ `M` fixed).
  5. Reads: `S(x) = B(x)ᵀ c`, where `B(x)` is the (sparse) spline basis row at query `x`. The compression stencil evaluates `g(·, S(·))` at its stencil points through the same reconstruction.
- **Births:** new characteristics are introduced at the boundary `x_b` (the lowest position) on a schedule; `N(t)` grows mid-solve. The newborn's `ℓ` is set by the production birth-flux formula, which reads `S(x_b)`.
- **Two-pass structure (already exists):** pass 1 runs the adaptive solver in plain `double` and *records* the schedule — step sizes, RK stage pattern, birth times/counts, insertion decisions. Pass 2 replays that schedule *frozen*, with no adaptive branching. The gradient work in this spec happens entirely in pass 2.
- **Functionals to differentiate:** first moments of the terminal state, `M = Σᵢ φ(xᵢ(T)) · e^{ℓᵢ(T)} · Δxᵢ(T)`, and time-integrated variants (per-step accumulations of the same form). Several functionals may be differentiated per solve.
- **Sizes:** `N ≈ 100–1000`, `k ≈ 17`, RK stages `s = 2–7`, steps `10³–10⁵`.

**Symbol table.**

| symbol | meaning |
|---|---|
| `xᵢ, ℓᵢ` | characteristic position, log-density (the taped ODE state) |
| `wⱼ` | quadrature weight `e^{ℓⱼ}Δxⱼ` (derived, taped) |
| `z_m, A_m, s_m` | knot positions (fixed), knot aggregates, knot samples `exp(A_m)` |
| `M, c, B(x), B′(x)` | collocation matrix (fixed), spline coefficients, basis row and its derivative at query `x` |
| `Sᵢ` | field value read `S(xᵢ) = B(xᵢ)ᵀc` |
| `Dᵢ` | compression value: production FD secant of `∂ₓ[g(x,S(x))]` at `xᵢ` |
| `D̃ᵢ` | geometric compression: neighbor-velocity secant (defined in §4) |
| `θ, M` | parameters; a moment functional |
| overbar (`x̄`, `Ā`, …) | reverse-mode adjoint of that quantity |

## 2. The problem this layer solves

**Symptom.** Instantiating the solver on an AD scalar type and taping pass 2 naively produces a gradient that is *internally consistent* (forward-mode JVP equals reverse-mode VJP to machine precision) and *value-exact* (bit-identical trajectory), yet disagrees with a converged finite difference of the model-as-run by an **O(1) factor** — observed up to ~2–3×, with sign errors in variants — on any functional, i.e. on all of them, because every `ℓᵢ` integrates the compression `Dᵢ`. The disagreement does not shrink with the FD step. The failure is localized entirely in the θ-linearization of `Dᵢ`.

**Root cause (two mechanisms).**

1. **Compression is rendered twice, by two different discrete operators.** Define the per-characteristic mass `mᵢ = e^{ℓᵢ} · Δxᵢ`. In the continuum, `dmᵢ/dt = −r·mᵢ`: the compression in `ℓ`'s ODE cancels exactly against the geometric compression of the spacings (`d(Δxᵢ)/dt` = velocity differences). Every moment consumes only `mᵢ`, so every moment depends on the *cancellation*, not on either rendering. In the discrete solver the two renderings — the spline-secant `Dᵢ` and the spacing dynamics — cancel only to truncation error. That is harmless for **values**. It is fatal for **derivatives**: the two operators respond differently to a θ-perturbation, so their θ-linearizations do *not* cancel, and the tape faithfully differentiates the mismatch, which is O(1) relative to the surviving signal.
2. **The spectral secant is the wrong object to linearize at all.** `Dᵢ` differentiates a rank-17 reconstruction in space, at a moving query that is itself a *source* of the field it reads. The θ-derivative of that construction is dominated by sub-grid behavior — how each source's individual imprint slides across the fixed knot basis, including the reader's own imprint (a self-force) — rather than by the macroscopic field. No pointwise repair of `d(Dᵢ)/dθ` (freezing channels, un-freezing channels, analytic self-subtraction at the read point, deposit-level leave-one-out) has been found that is simultaneously correct for trajectory-moving parameters and for coupling-only parameters; each fixes one class and breaks the other.

**Hard constraints on the solution.**

- **C-VALUES:** enabling AD must not change the computed trajectory. The taped pass reproduces the `double` pass bit-for-bit. Any substitution may change a *derivative*, never a *value*.
- **C-STATE:** the state stays `(x, ℓ)`. Existing infrastructure integrates log-density; switching the transported variable to conserved mass is out of scope.
- **C-FRAMEWORK:** the AD engine is XAD's general tape. No custom AD engine.
- **C-COST:** one gradient (all `|θ|` parameters) costs O(1) forward-solve equivalents; tape memory is manageable at the stated sizes (see P6 for the budget).
- **C-BOTH-CLASSES:** the gradient is correct simultaneously for *trajectory-moving* parameters (dominant effect through `dxᵢ/dθ`) and *coupling-only* parameters (whole effect through the field; includes a null-channel parameter whose true sensitivity is small, so any spurious term appears at full magnitude).

## 3. Structure we leverage

Everything in Part II follows from five facts about this specific system.

1. **XAD external-function callbacks.** XAD's tape supports user-inserted nodes (`CheckpointCallback` / external functions) whose *forward* action writes arbitrary values to registered outputs and whose *reverse* action (`computeAdjoint`) applies a hand-written adjoint to registered input/output slots. This is the licensed mechanism for (a) compressing a whole sub-computation to one node with an analytic transpose, and (b) **value-preserving derivative substitution**: outputs carry exact production values while the reverse rule linearizes a chosen (different) model.
2. **The geometric rendering of compression is already on the tape.** Neighbor spacings are live taped positions; velocity differences of taped reads are one subtraction away. So the *geometric* compression `D̃ᵢ = [g(x_{i+1},S_{i+1}) − g(x_{i−1},S_{i−1})] / (x_{i+1} − x_{i−1})` has a smooth, well-conditioned, closed-form linearization in terms of quantities the tape already carries. Substituting *its* linearization at the compression node makes the two renderings of compression cancel **exactly in the linearization** of every `mᵢ = e^{ℓᵢ}Δxᵢ` — restoring, at derivative level, the cancellation the continuum guarantees and the value level already has to truncation error.
3. **The one-sided kernel with vanishing diagonal.** `κ(z, z) = 0` means a source crossing a knot enters that knot's sum at value zero — the deposit is *continuous* in θ at membership changes. And because every characteristic obeys the same scalar ODE, trajectories cannot cross: the ordering, and hence every membership set `{j : xⱼ ≥ z_m}` evaluated at recorded positions, is stable data. There is no set-flip discontinuity for the adjoint to fight.
4. **The frozen schedule.** Steps, stages, births, and insertions are recorded data, not decisions. Pass 2 contains no branching on active values, the whole tape layout is knowable before recording (preallocation), replay is deterministic (checkpointing-safe), and "finite difference of the model-as-run" is a well-defined reference: rerun the frozen schedule at `θ ± ε`.
5. **The field pipeline is linear in the knot samples with fixed matrices.** For fixed `(x, w)`, the map `s ↦ (reads)` is `S(x) = B(x)ᵀ M⁻¹ s` with `B`, `M` fixed-structure. Its exact transpose is: gather reader adjoints into coefficient space, one `k×k` transpose solve, scatter through deposit rows — `O(N·k)` work, `O(N + k)` stored state per stage. The full stage Jacobian never needs to be formed or taped.

## 4. Design summary

**The commitment (the one decision everything else follows from):** *every derivative that touches the coupling flows through exactly two callback seams per RK stage — the field pipeline node and the compression node — whose forward outputs are the production `double` values and whose reverse rules are hand-written; everything else on the tape is plain scalar arithmetic.* This is kept true by structure: the field pipeline and the compression stencil are only reachable through the two callback classes; there is no other code path by which an active variable enters them.

```
        x[·], ℓ[·]                     (taped ODE state, preallocated at N_max)
            │
   taped:  w_j = exp(ℓ_j)·Δx_j         (production quadrature, plain AReal ops)
            │
   ┌────────▼──────────────────────────┐
   │ CB-1  CouplingFieldOp  (callback) │  forward: production double pipeline,
   │ in : x[1..N], w[1..N], θ          │           outputs bit-exact
   │ out: S_i = S(x_i), i=1..N;        │  reverse: exact analytic transpose of
   │      S_b = S(x_b) on birth steps  │           the literal pipeline map
   └────────┬──────────────────────────┘
            │ S_i
   ┌────────▼──────────────────────────┐
   │ CB-2  CompressionOp    (callback) │  forward: D_i = production spectral
   │ in : x_{i−1}, x_{i+1},            │           secant (bit-exact values)
   │      S_{i−1}, S_{i+1}, θ          │  reverse: linearization of the
   │ out: D_i, i=1..N                  │           geometric secant D̃_i
   └────────┬──────────────────────────┘
            │ D_i
   taped:  dx_i/dt = g(x_i, S_i; θ)                (plain AReal, closed form)
           dℓ_i/dt = −(D_i + r(x_i, S_i; θ))       (plain AReal, closed form)
           RK stage combination; moment accumulators; birth map at recorded times
```

**Which map is the returned gradient the derivative of — and why that is the right one.** The gradient is the exact discrete adjoint of the **geometric-compression model** — the frozen-schedule solver identical to production except that the compression value in `ℓ`'s ODE is `D̃ᵢ` instead of `Dᵢ` — linearized along the recorded production trajectory. This is the right target for the functionals at hand because (a) the continuum sensitivity of a moment contains no compression term at all (integrate the PDE by parts: `dM/dt = ∫(φ′g − φr)n dx + boundary flux`); (b) in the discrete model every moment consumes `mᵢ = e^{ℓᵢ}Δxᵢ`, in which compression appears twice and must cancel — and the geometric pairing preserves that cancellation *exactly in the linearization*, while the spectral pairing (the naive tape) does not; (c) the geometric and production models differ by the secant gap `Dᵢ − D̃ᵢ`, a truncation-order quantity that is computed and monitored on every run.

**Approximation ledger.** Every place the derivative is not the literal derivative of the production code, named and monitored:

| id | approximation | bound | diagnostic (always on) |
|---|---|---|---|
| **A1** | Compression derivative substitution: value = spectral secant `Dᵢ`, linearization = geometric secant `D̃ᵢ`. | Gradient bias is controlled by the secant gap; proxy bound `Γ = Σ_steps h·maxᵢ|Dᵢ − D̃ᵢ|` on the induced `ℓ`-trajectory discrepancy. Shrinks with spatial/temporal refinement of the production solve. | Per-stage `max`/`rms` of `|Dᵢ − D̃ᵢ|`; running `Γ`; configurable alarm threshold. (Test V4a/V4b isolates the end-to-end effect.) |
| **A2** | Frozen schedule: the gradient is of the fixed-schedule model; the schedule's own θ-sensitivity is dropped. | Estimated by gradient convergence under pass-1 tolerance refinement (compute the gradient at 2–3 recording tolerances; the spread bounds A2). | Tolerance-refinement study, run per model configuration (V6). |
| **A3** (conditional) | Deposit-derivative kink: `∂A_m/∂xⱼ` jumps by `∂ₓκ(z_m, z_m)·wⱼ` as source `j` crosses knot `m` (the *value* is continuous since `κ(z,z)=0`). Only material if `∂ₓκ(z,z) ≠ 0`. | Per-event magnitude `O(∂ₓκ(z,z)·wⱼ)`; `O(k/N)` in aggregate. | Near-knot counter: per stage, count sources with `|xⱼ − z_m| < δ_knot`. Contingency (only if V4/V5 scatter correlates with the counter): a C¹ ramp of the cutoff over a sub-knot width — a *value-changing* forward modification, to be signed off separately. |

No self-exclusion, cavity, or masking appears anywhere in this design. The exact transpose of the literal pipeline (CB-1) includes each characteristic's own contribution to its own read; that self-coupling is genuine structure of the model-as-run and is kept. The pathological self-force lived exclusively in the θ-derivative of the *spectral slope construction*, which A1 removes from the derivative path.

**Future options behind the same seam (out of scope, listed so nobody designs against them):** replacing CB-1's interior transpose with an exact pairwise one-sided sum (O(N²), reconstruction-free) or a structured/separable-kernel recurrence; adding a companion slope field (`∂_zκ`-aggregate) if `g` or `r` ever read `∂ₓS` directly; a conserved-mass twin model for cross-checking. All are drop-in replacements *inside* the two callbacks; none changes the tape architecture.

---

# Part II — Implementation plan

Phases P0–P5 are sequential; each has an acceptance gate. P6 is conditional on measured memory. Do not start a phase before the previous phase's gate passes.

Notation for code sketches: `Real` is the templated scalar (`double` on pass 1, `xad::AReal<double>` on pass 2); `passive(v)` extracts the `double` value of an active scalar (XAD: `xad::value(v)`). All XAD API names used below must be verified against your installed version — see Appendix B before writing any callback code.

## P0 — Prerequisites

**P0.1 — Scalar templating.** Template the pass-2 code path on the scalar type: rates `g`, `r`, their closed forms, the quadrature-weight expression `wⱼ = exp(ℓⱼ)·Δxⱼ`, the RK stage combination, the moment accumulators, and the birth formula. The field pipeline and the compression stencil are *not* templated — they remain `double`-only and are reached exclusively through the callbacks (this enforces the commitment in §4). Provide, in one header, the closed-form partials of `g` needed by CB-2: `g_x(x,S;θ)`, `g_S(x,S;θ)`, `g_θp(x,S;θ)` for each parameter `p`. Unit-test each partial against central FD of `g` at 10 random points, relative tolerance 1e-6.

**P0.2 — Schedule record.** A plain struct written by pass 1 and read by pass 2, containing: ordered accepted step sizes `h_n`; the RK tableau id; for each step, the list of birth events (count, and the recorded inputs to the birth formula); insertion events likewise; total step count; final `N`. Pass 2 must consume *only* this struct for all control flow — grep the pass-2 loop for any comparison involving an active value and fail the build if one exists (a `static_assert`-style trait check on comparison operators of `Real` in the driver translation unit is a practical enforcement).

**P0.3 — Preallocation.** Allocate all per-characteristic arrays (`x`, `ℓ`, stage buffers, adjoint seeds) at `N_max` (known from pass 1) as structure-of-arrays, before recording starts. The active count `N(t)` is a fill pointer advanced at recorded birth steps. **Never resize a container of `AReal` mid-recording.**

**P0.4 — Standing assertions** (both passes, every accepted step; fail hard with the step index):
- *Order invariance:* `x` strictly increasing over the active prefix.
- *Bitwise replay:* FNV-1a (or equivalent) hash over the raw byte images of `x[0..N)`, `ℓ[0..N)`; pass 2 (hashing `passive(...)` values) must equal pass 1's recorded hash. Any mismatch means a value leak — a code-path or FP-contraction difference between the `double` and `AReal` instantiations. If mismatches trace to compiler FMA/vectorization differences, align flags (e.g. `-ffp-contract=off` on both instantiations) rather than loosening the assertion.

**P0.5 — θ registration.** `θ` is a fixed-size array of `AReal` registered as tape inputs once, before the step loop. Both callbacks capture the θ slots; taped closed forms consume the `AReal` θ directly.

**Gate P0:** pass 2 with `Real = AReal` but *no callbacks yet* (field pipeline still called on passive values, outputs re-registered naively — gradients will be wrong, that's expected) runs the full frozen schedule with all P0.4 assertions green. This certifies templating and replay before any adjoint work.

## P1 — CB-1: `CouplingFieldOp` (one instance per RK stage)

### Interface

- **Active inputs (slots captured):** `x[0..N)`, `w[0..N)`, `θ[0..P)`.
- **Active outputs (fresh actives, values set by the callback):** `S_out[0..N)` = value reads at all particle positions; plus `S_b` = read at the fixed boundary `x_b` on steps where the birth map will consume it.
- **Not outputs:** anything the compression stencil needs at sub-grid stencil points — those are evaluated *passively inside CB-2* from CB-1's stashed reconstruction.

### Forward (during recording)

1. Extract passive `x̂ⱼ = passive(xⱼ)`, `ŵⱼ = passive(wⱼ)`, `θ̂`.
2. Call the **unmodified production pipeline** on these doubles: aggregates `A_m`, samples `s_m = exp(A_m)`, coefficients `c = M⁻¹s` (reuse the one-time factorization of `M`, P1-note below), reads `Ŝᵢ = B(x̂ᵢ)ᵀc` and `Ŝ_b`.
3. Write `Ŝᵢ` into the output actives **exactly** (no arithmetic detour — set the value, do not compute it via active ops), and register the outputs with the tape per the XAD external-function idiom.
4. **Stash** (owned by the callback object, one per stage): `x̂[0..N)`, `ŵ[0..N)`, `A[0..k)`, `c[0..k)`, the active count `N`, and the input/output slot lists. Do **not** stash basis rows or κ rows — recompute them in reverse (they are `O(stencil)` closed-form evaluations; recomputation is cheaper than the memory traffic).
5. Insert the callback into the tape (`tape.insertCallback(this)` or version equivalent).

### Reverse (`computeAdjoint`) — the exact transpose of the literal pipeline

Let `S̄ᵢ` be the fetched-and-reset output adjoints (`getAndResetOutputAdjoint` on each output slot; the reset matters — see Appendix B). Then, in order:

```
(F1)  coefficient-space gather:   c̄ = Σᵢ S̄ᵢ · B(x̂ᵢ)   (+ S̄_b · B(x_b))
      [B(x̂) has O(stencil) nonzeros; O(N·stencil) work]

(F2)  knot-sample cotangent:      s̄ = M⁻ᵀ c̄
      [one k×k transpose solve on the stored factorization]

(F3)  aggregate cotangent:        Ā_m = s_m · s̄_m        (since s_m = exp(A_m);
      for a general pointwise map ψ:  Ā_m = ψ′(A_m)·s̄_m, and θ̄ += ∂θψ(A_m)·s̄_m)

(F4)  query-motion channel:       x̄ᵢ += S̄ᵢ · ( B′(x̂ᵢ)ᵀ c )
      [the reconstruction slope at the reader, used as a stored value coefficient;
       NO such term for S_b — its query x_b is a constant]

(F5)  deposit scatter, for each source j and each knot m with z_m ≤ x̂ⱼ
      (intersected with κ's support, if compact):
          w̄ⱼ += Ā_m · κ (z_m, x̂ⱼ; θ̂)
          x̄ⱼ += Ā_m · ∂ₓκ(z_m, x̂ⱼ; θ̂) · ŵⱼ
          θ̄ₚ += Ā_m · ∂θₚκ(z_m, x̂ⱼ; θ̂) · ŵⱼ
      [membership evaluated at recorded positions — stable data, per §3.3]

(F6)  incrementAdjoint on every x, w, θ input slot with the accumulated values.
```

Complexity per stage: `O(N·(stencil + k_support) + k²)` flops, `O(N + k)` stash. Two channels are deliberately both present: the *knot channel* (F1–F3, F5) and the *query channel* (F4). Their `j = i` entries — the reader's own deposit and its own query motion — are part of the exact Jacobian and are **not** masked.

### Edge cases (must mirror production exactly)

- **Extrapolation:** if any read query falls outside `[z₁, z_k]`, the transpose must use the basis rows of production's extrapolation/clamping rule (a clamped read has `B′ = 0` outside — then F4 contributes nothing there). Copy the rule from the production evaluator; do not invent one.
- **Empty knots:** knots with no sources above them (`A_m = 0`) participate normally; `s_m = 1` — no special case.
- **Endpoint quadrature:** `w` uses production's endpoint spacing rule; that lives in taped code (P0.1), not in this callback, but F5's `ŵⱼ` must be the same numbers — guaranteed by stashing `passive(w)`.

### Sketch

```cpp
struct CouplingFieldOp : xad::CheckpointCallback<Tape> {          // verify base class name
  int N; StageId id;
  std::vector<Slot> xin, win, thin, sout; Slot sb_out{invalid};
  std::vector<double> xs, ws, A, c;                               // the stash
  const SplineFactor& MF;                                         // shared, factored once

  void record(Tape& t, std::span<AReal> x, std::span<AReal> w,
              std::span<AReal> th, std::span<AReal> S_out, AReal* S_b);
              // extract passives, run production pipeline, set output values
              // exactly, capture slots, stash, t.insertCallback(this)

  void computeAdjoint(Tape* t) override {
    std::vector<double> Sbar(N);
    for (int i=0;i<N;++i) Sbar[i] = t->getAndResetOutputAdjoint(sout[i]);
    // F1..F5 as above, accumulating xbar[], wbar[], thbar[]
    for (int i=0;i<N;++i) t->incrementAdjoint(xin[i], xbar[i]);
    for (int i=0;i<N;++i) t->incrementAdjoint(win[i], wbar[i]);
    for (int p=0;p<P;++p) t->incrementAdjoint(thin[p], thbar[p]);
  }
};
```

**P1-note (factorization reuse):** factor `M` once at startup (LU or banded solve appropriate to the spline family); use it for every forward solve (`M⁻¹s`) and every reverse transpose solve (`M⁻ᵀc̄`). Knots are fixed for the lifetime of the run.

### Unit tests U1 (small config: `N = 7`, `k = 5`, random smooth `κ`, random states)

- **U1.a — value parity:** callback outputs equal the direct production pipeline bitwise.
- **U1.b — Jacobian correctness:** implement a small dense JVP of the same pipeline (directional derivative of F1–F5's forward map — share the formulas, templated) and check against central FD of the double pipeline, tol 1e-6 relative. This validates the *formulas*.
- **U1.c — adjoint identity:** for 20 random tangent/cotangent pairs `(v, u)`: `⟨u, Jv⟩` from the JVP vs `⟨Jᵀu, v⟩` from `computeAdjoint` run standalone; agree to 1e-12 relative. This validates the *transpose*.
- **U1.d — slot mechanics:** record a 3-op toy tape around the callback, seed, sweep, confirm adjoints land on the right slots and repeated sweeps don't double-count (the reset semantics).

**Gate P1:** U1 all green.

## P2 — CB-2: `CompressionOp` (one instance per RK stage, vectorized over `i`)

### Interface

- **Active inputs:** the neighbor positions and neighbor field reads each `Dᵢ` depends on under the geometric model — interior: `x_{i−1}, x_{i+1}, S_{i−1}, S_{i+1}`; endpoints: the one-sided pair including the point's own `(xᵢ, Sᵢ)` — plus `θ`.
- **Active outputs:** `D_out[0..N)`.

### Forward

1. Compute `D̂ᵢ` by the **unmodified production spectral stencil**, passively, from CB-1's stashed reconstruction `(c)` — call the same double routine production uses (it may evaluate `g(·, B(·)ᵀc)` at sub-grid stencil points; all passive).
2. Write `D̂ᵢ` into the output actives exactly; capture slots; stash `x̂_{i±1}, Ŝ_{i±1}` (the exact primals the reverse rule will linearize at — stash them rather than re-deriving, so value/derivative primals cannot drift apart), and also compute and stash the geometric values `D̃ᵢ` for the diagnostic.
3. **Diagnostic D1 (always on):** per stage, record `maxᵢ|D̂ᵢ − D̃ᵢ|`, `rms`, and accumulate `Γ += h · maxᵢ|D̂ᵢ − D̃ᵢ|` over the run. Emit to the run log; alarm above the configured threshold. This is approximation A1's monitor.

### Reverse — linearization of the geometric secant

Interior `i`, with `Δᵢ = x̂_{i+1} − x̂_{i−1}`, `g⁺ = g(x̂_{i+1}, Ŝ_{i+1}; θ̂)`, `g⁻` likewise, `D̃ᵢ = (g⁺ − g⁻)/Δᵢ` (recompute from stash), and fetched output adjoint `D̄ᵢ`:

```
(F7)  x̄_{i+1} += D̄ᵢ · ( g_x⁺ − D̃ᵢ ) / Δᵢ
      x̄_{i−1} += D̄ᵢ · ( D̃ᵢ − g_x⁻ ) / Δᵢ
(F8)  S̄_{i+1} += D̄ᵢ ·   g_S⁺        / Δᵢ
      S̄_{i−1} += D̄ᵢ · ( − g_S⁻ )     / Δᵢ
(F9)  θ̄ₚ      += D̄ᵢ · ( g_θₚ⁺ − g_θₚ⁻ ) / Δᵢ
```

Endpoints (`i = 1` uses pair `(1,2)`; `i = N` uses `(N−1, N)`): same formulas with the one-sided pair, the point's *own* `(x̄ᵢ, S̄ᵢ)` receiving the corresponding entries. Mirror the *shape* of production's endpoint stencil (if production uses a different one-sided form at boundaries, use its geometric analogue — document the choice in code). A newborn's first stage after birth is an ordinary `i = 1` endpoint.

The `S̄_{i±1}` increments land on CB-1's output slots; because CB-2 is inserted *after* CB-1 within the stage, the tape's reverse order runs CB-2's `computeAdjoint` first, and CB-1 then transposes those contributions onward. **Verify the LIFO callback ordering guarantee in your XAD version (Appendix B item 6) — this composition depends on it.**

Everything in F7–F9 is a first derivative of a smooth closed form at stashed primals. Nothing sub-grid is differentiated; nothing large must cancel.

### Unit tests U2

- **U2.a — value parity:** `D_out` equals the production stencil bitwise on random configs.
- **U2.b — derivative-model correctness:** F7–F9 against central FD **of the `D̃` formula** (not of `D̂` — they differ by design; that difference is A1), tol 1e-8 relative.
- **U2.c — adjoint identity** on the `D̃` map, as U1.c.
- **U2.d — endpoint cases:** U2.b/c repeated for `i = 1`, `i = N`, and `N = 2`.

**Gate P2:** U2 green, and the D1 log shows the secant gap at the expected truncation magnitude on a nominal run (record the baseline number; it is the reference for all future regressions).

## P3 — Birth map (recorded events, explicit taped map)

At each recorded birth step, after CB-1:

1. Advance the fill pointer; the newborn slot's position is the constant `x_b` (register as a taped constant or an input with zero seed — it carries no θ-dependence unless production says otherwise).
2. Set the newborn's `ℓ` by the **production birth formula**, written in taped `Real` arithmetic, consuming: the recorded schedule inputs (plain data), the active read `S_b` from CB-1's output, and `θ`. This is ordinary smooth taped code — the reverse sweep automatically routes the newborn's adjoint into the pre-birth field (through `S̄_b` → CB-1) and into `θ`.
3. Insertion events (if the schedule contains them) follow the same pattern: position/`ℓ` of the inserted characteristic computed by production's interpolation formula in taped arithmetic from neighbor states; the *decision* (when/where) is frozen data.

**Tests U3:** (a) value parity of the birth step against pass 1 (covered by the P0.4 hash, but assert locally too for a sharp failure point); (b) a *causality probe*: a synthetic parameter that enters only a characteristic's post-birth dynamics must show exactly zero adjoint at pre-birth times (checked by inspecting `θ̄` contributions per step in a debug sweep).

**Gate P3:** full pass-2 recording of a nominal run completes with callbacks and births, P0.4 assertions green.

## P4 — Driver assembly and the reverse sweep

Pass-2 recording loop (per accepted step `n`, per stage `s`):

```
compute stage state (taped RK combination from tableau + recorded h_n)
compute w[0..N) = exp(ℓ)·Δx           // taped, production quadrature formula
CB1[n][s].record(tape, x, w, θ, S, &S_b)
CB2[n][s].record(tape, x, S, θ, D)    // production values, geometric reverse
k_s: dx_i = g(x_i, S_i; θ)            // taped closed form
     dℓ_i = −(D_i + r(x_i, S_i; θ))   // taped closed form
end stages; combine stages into (x, ℓ) update      // taped
apply recorded birth/insertion events (P3)
update time-integrated accumulators:  M_acc += h_n · Σ ρ(x_i, S_i; θ)·exp(ℓ_i)·Δx_i
assert order invariance + bitwise hash (P0.4)
```

After the loop: form each functional `M` as taped arithmetic over the terminal state and/or accumulators; `tape.registerOutput(M)`.

Reverse sweep, per functional: clear adjoints, seed `derivative(M) = 1`, `tape.computeAdjoints()`, harvest `dM/dθₚ = derivative(θₚ)`. Multiple functionals = one sweep each over the same tape (clear adjoints between sweeps — and note the callbacks' `getAndResetOutputAdjoint` idiom is what makes repeated sweeps safe). Cost per sweep is comparable to one forward pass; total gradient cost is O(1) forward-equivalents per functional, independent of `|θ|` (constraint C-COST).

**Callback lifetime:** the tape must be able to invoke `computeAdjoint` after recording ends, possibly several times. Own the callbacks in a registry that outlives all sweeps; free them only after the last sweep (or per your XAD version's documented ownership rule — Appendix B item 7).

**Gate P4:** a full record + reverse sweep runs end-to-end; `θ̄` is finite and nonzero; a parameter that provably enters nothing (a dummy registered input) returns exactly zero.

## P5 — Verification suite

Run V1–V3 in CI on a small config every commit; V4–V7 per model configuration and per release.

- **V1 — unit adjoint identities.** U1.c, U2.c as CI tests (already built in P1/P2).
- **V2 — bitwise replay.** The P0.4 hash assertion over a full nominal run. Zero tolerance.
- **V3 — order invariance.** P0.4, both passes. Zero tolerance.
- **V4 — whole-model gradient vs finite differences.** Protocol:
  1. Record the schedule at base `θ₀` (pass 1). All FD runs below **replay this same frozen schedule** — the reference is the fixed-schedule model, matching what the adjoint differentiates (net of A1).
  2. For each parameter `p` and each `ε` in a log-spaced sweep (e.g. `1e−3 … 1e−7` × a per-parameter scale): plain-double replays at `θ₀ ± ε eₚ`; central difference. Identify the plateau (window where FD varies < 0.1% between adjacent `ε`); the plateau value ± spread is the reference.
  3. **V4a (exact target, machine-checkable):** run the solver with the test-only flag `--compression=geometric`, which uses `D̃ᵢ` **as the value** in `ℓ`'s ODE. The adjoint of that model has *no* A1 gap. Acceptance: `|adjoint − FD| ≤ max(0.2%, 3×plateau spread)` for **every** parameter — trajectory-moving and coupling-only alike, simultaneously. Any failure here is a bug, not an approximation.
  4. **V4b (production values):** same comparison with the production flag. Acceptance: ≤ 1% relative on all parameters. The V4b−V4a discrepancy is A1 end-to-end; confirm it correlates with the D1 gap `Γ` and shrinks under forward refinement (halve the recording tolerance; the discrepancy should drop at truncation order).
- **V5 — null-channel and decoupled probes.**
  - *Coupling-only probe:* a parameter entering **only** `κ` (e.g. a global κ-scale). Its whole sensitivity flows through the field; any spurious or dropped term in CB-1 appears at full magnitude and sign. Acceptance: within the V4 plateau spread, using a mixed tolerance `|ad − fd| ≤ 0.01·‖∇M‖₂ + 0.05·|fd|` (the absolute term matters because the true value can be small).
  - *Decoupled probe:* a parameter feeding only an accumulator that `M` does not consume. Acceptance: `|ad| ≤ 1e−12 · ‖∇M‖₂` (exact-zero channel).
- **V6 — schedule-sensitivity study (bounds A2).** Compute the V4b gradient at 2–3 pass-1 recording tolerances; report the spread as the A2 bound for this configuration. If the spread exceeds the accuracy the downstream consumer needs, tighten the recording tolerance — do not attempt to differentiate the schedule.
- **V7 — mechanical A/B.** A flag `--coupling=taped` reroutes CB-1's *interior* through plain `AReal` operations (same mathematics, tape does the transpose) with CB-2 unchanged. On a small config, the two gradients must agree to ~1e−10 relative — they are two mechanizations of the *same* linearization. This is the standing regression localizer: a future discrepancy means the analytic transpose (F1–F6) drifted from the pipeline; agreement means any V4 failure lies elsewhere. (Do not run at full size — the taped variant's memory is enormous by design.)

**Gate P5 (definition of correct):** V1–V5 green; V6 documented; V7 green at small config.

## P6 — Memory and performance (conditional)

**Measure before engineering.** Record 10 representative steps at target `N`; read `tape.getMemory()` plus the callbacks' stash total; extrapolate linearly to the full run:

```
Mem_total ≈ steps × [ ops_per_stage × stages × bytes_per_tape_entry
                      + (2N + 2k)·8 × stages   (callback stashes)
                      + per-step taped state ]
```

The dominant term is taped rate arithmetic: roughly 30–100 ops/particle/stage. At `N = 100`, 4 stages, 2·10³ steps this lands near ~1 GB (acceptable); at `N = 1000`, 4 stages, 2·10⁴ steps it extrapolates to the 10²-GB range — **segment checkpointing is then required, not optional.** Decide from the measurement, against your actual RAM budget (rule of thumb: proceed uncheckpointed only below ~25% of available RAM).

**Segment checkpointing (XAD's documented pattern; verify against your version's checkpointing chapter):**

1. Partition the step sequence at **joints = every birth/insertion event, and at most K steps apart** (K from the memory formula). Every segment then has constant `N` and constant memory layout.
2. During recording, do **not** tape segment interiors. At each joint, snapshot the passive state `(x, ℓ, accumulators)` and insert one segment-callback holding the snapshot and the segment's slice of the recorded schedule; register the segment's outputs (the state at the next joint) as the tape-visible actives.
3. In the segment-callback's `computeAdjoint`: re-record the segment **actively** on the tape after the current position (this nested recording contains the full P4 stage loop, including fresh CB-1/CB-2 instances — they nest cleanly); seed the stored output adjoints; compute adjoints over the nested range; harvest input-state and θ adjoints; reset the tape position to reclaim the nested memory.
4. Cost: one extra forward evaluation of each segment during the sweep (≈ 2× total flops); memory drops to `O(max segment) + O(joints × N)` snapshots.
5. Re-run the **entire** V-suite with checkpointing enabled at small scale before enabling it by default; V2's per-step hash inside nested re-recordings doubles as the determinism guarantee this pattern relies on.

**Other performance rules (unconditional):** structure-of-arrays throughout; factor `M` once; recompute basis/κ rows in reverse rather than stashing them; do not parallelize the sweep at these sizes; do not cache Jacobians across stages (that would be an unnamed approximation).

## P7 — Rollout and definition of done

**Flags:** `--grad=off|on`; `--compression=production|geometric` (test-only value model for V4a); `--coupling=callback|taped` (V7); `--checkpoint=off|segments`; diagnostic thresholds for D1 and the A3 near-knot counter.

**Logging per gradient run (always):** D1 stats and `Γ`; A3 counter; per-step hash status; tape memory; wall time forward vs sweep.

**Definition of done:**
1. Gates P0–P5 green; P6 decision documented with the measured numbers.
2. V4a ≤ 0.2%/spread on all parameters; V4b ≤ 1%; V5 probes green.
3. A1/A2 numbers for the nominal configuration recorded in the run report (the approximation ledger is *populated*, not just defined).
4. V7 A/B wired into CI at small config.
5. No masking/cavity/self-exclusion code anywhere; no active-value branching in pass 2 (enforced per P0.2).

---

# Appendix A — Formula sheet (single source of truth)

Forward (production, values):
```
w_j  = exp(ℓ_j)·Δx_j                          (production quadrature Δx rule)
A_m  = Σ_{j: x_j ≥ z_m} κ(z_m, x_j; θ)·w_j
s_m  = exp(A_m);   c = M⁻¹ s;   S(x) = B(x)ᵀc;   S′(x) = B′(x)ᵀc
D_i  = production spectral secant of ∂ₓ[g(x, S(x))] at x_i   (value only)
D̃_i  = [g(x_{i+1},S_{i+1};θ) − g(x_{i−1},S_{i−1};θ)] / (x_{i+1} − x_{i−1})
dx_i/dt = g(x_i,S_i;θ);   dℓ_i/dt = −(D_i + r(x_i,S_i;θ))
M    = Σ φ(x_i)·exp(ℓ_i)·Δx_i   (+ time-integrated accumulators)
```

CB-1 reverse (exact transpose; F1–F6): `c̄ = Σᵢ S̄ᵢB(x̂ᵢ) (+S̄_b B(x_b))`; `s̄ = M⁻ᵀc̄`; `Ā_m = s_m s̄_m`; `x̄ᵢ += S̄ᵢ·B′(x̂ᵢ)ᵀc`; for `z_m ≤ x̂ⱼ`: `w̄ⱼ += Ā_m κ`, `x̄ⱼ += Ā_m ∂ₓκ·ŵⱼ`, `θ̄ₚ += Ā_m ∂θₚκ·ŵⱼ`.

CB-2 reverse (geometric linearization; F7–F9): with `Δᵢ = x̂_{i+1}−x̂_{i−1}`: `x̄_{i+1} += D̄ᵢ(g_x⁺−D̃ᵢ)/Δᵢ`; `x̄_{i−1} += D̄ᵢ(D̃ᵢ−g_x⁻)/Δᵢ`; `S̄_{i±1} += ±D̄ᵢ g_S^±/Δᵢ`; `θ̄ₚ += D̄ᵢ(g_θₚ⁺−g_θₚ⁻)/Δᵢ`. Endpoints: one-sided pair, own-point terms included.

Diagnostics: `gap_i = D_i − D̃_i`; `Γ = Σ_steps h·maxᵢ|gap_i|` (A1). Near-knot counter: `#{(j,m): |x̂ⱼ−z_m| < δ_knot}` per stage (A3).

# Appendix B — XAD API verification checklist (do this before P1)

Confirm each against your installed XAD version's documentation (external functions / checkpointing chapters); record the exact names in the code as constants of the wrapper layer:

1. Callback base class and virtual (`CheckpointCallback<Tape>` / `computeAdjoint(Tape*)`).
2. How to obtain a **slot/handle** for an existing active variable (inputs) — and whether inputs must be explicitly registered.
3. How to create callback **outputs**: fresh actives whose values are set without recording arithmetic, registered so downstream taped ops depend on them.
4. `getAndResetOutputAdjoint(slot)` — confirm the reset semantics (required for multi-sweep safety).
5. `incrementAdjoint(slot, double)` for inputs.
6. **Callback invocation order during the sweep is reverse insertion order (LIFO).** CB-2 → CB-1 composition depends on this; write a two-callback toy test that fails loudly if it does not hold.
7. Callback ownership/lifetime relative to `computeAdjoints()` and tape destruction; whether the tape deletes callbacks.
8. Position/rewind API for nested re-recording (P6): get position, compute adjoints over a range, reset to position.
9. Forward-mode type (`FReal`) availability — used only in unit tests (U1.b/U2.b share the templated JVP formulas); callbacks are reverse-only.
10. Behavior of `xad::value()` on actives inside a recording (must not record).

# Appendix C — Consolidated pitfalls

1. **Never** branch on an active value in pass 2; the schedule struct is the only source of control flow (P0.2 enforcement).
2. **Never** resize an `AReal` container mid-recording (P0.3).
3. Set callback output **values exactly** — assignment of the production double, not arithmetic that reproduces it.
4. Stash the primals the reverse rule linearizes at (`x̂_{i±1}, Ŝ_{i±1}`); recomputing them via a different path invites value/derivative drift — the exact disease this design exists to remove.
5. The taped `w` expression must be **bit-identical** to production's quadrature (same ops, same order); V2 catches violations, but write it by copying the production expression, not by re-deriving it.
6. FD references must **replay the frozen schedule**; FD with re-adaptation measures a different model (its gap to V4b is A2, not an error).
7. Endpoint stencils: CB-2's derivative model must mirror the *shape* of production's boundary stencil; a mismatch shows up as V4 failures concentrated in parameters affecting boundary characteristics.
8. Don't forget `θ̄` accumulation in **both** callbacks (κ's θ in F5; g's θ in F9) — a missing θ-channel is exactly what the V5 coupling-only probe is designed to expose.
9. Compiler FP differences between `double` and `AReal` instantiations (FMA contraction, vectorization) break V2; align flags rather than weakening the assertion.
10. Clear adjoints between sweeps for multiple functionals; rely on the reset-on-fetch idiom inside callbacks.
11. If V4b degrades over time but V4a and V7 stay green, the *forward solve's* resolution degraded (A1 grew) — check D1/Γ before touching the gradient code.
