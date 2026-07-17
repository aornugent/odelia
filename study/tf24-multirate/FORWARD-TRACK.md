# TF24/TF24f forward multi-rate track — work order + state log

*Companion to [`tf24-multirate-implementation-plan.md`](./tf24-multirate-implementation-plan.md)
(the full MRI plan) and [`tf24-rodas-multirate.md`](./tf24-rodas-multirate.md) (the RODAS/multirate
evidence). This doc is the **forward-only** track — wiring MRI into the real TF24/TF24f patch — plus
a durable **state log** of everything measured so far, so nothing valuable is lost to a context wipe.*

---

## A. State log (what is established, and where)

All committed on branch `claude/tf24-multi-rate-stepper-n5audm` (plant-dev) and mirrored on
`aornugent/odelia` branch `tf24-multirate-study`; discussion on **aornugent/odelia#43** (reply to
**traitecoevo/odelia#38**).

**Measured facts (do not re-derive):**
1. **RODAS does not help the TF24 soil block.** Step collapse is **accuracy-limited, not stability-
   limited**: RK45 step count flat (3941→4137) across a **3000× drainage-stiffness sweep**; 100 % of
   steps accuracy-limited (`h·|λ| ~ 1e-3` vs stability ceiling ~3.3); RODAS ~1.5× more steps, ~5×
   slower. (`tf24-rodas-multirate.md`, `bench_main.R`, `diagnostic.R`.)
2. **Global dense RODAS is O(N³)** when the soil is coupled: ~470× slower than RK45 at N=205 (27 s vs
   57 ms), infeasible beyond. (`cost_sweep.R`.)
3. **Multi-rate is flat in the big-block size M** (~20–29 ms, N=55→805), ~1200× faster than global
   RODAS at N=205. (`multirate_runner.cpp`, `cost_sweep.R`.)
4. **MRI-GARK forward, validated** (`mri_core.hpp`, `test_mri.cpp`): collapse identities to machine
   precision (`f^F≡0`→base ERK 1e-16; `f^S≡0`→inner 1e-13); coupling order **1/2/2/3** (FwdEuler/
   Midpoint/Heun/Kutta3). On the soil+canopy surrogate at the **kink-aligned daily macro step**:
   Midpoint **5×**, Kutta3 **156×** more accurate than the Lie split at 1.44× cost. **Crossing
   forcing kinks (H>1) negates high order** → kink-aligned (daily) macro grid is a requirement.
   (`mri_soil_test.R`.)
5. **Forward stability across scenarios** (`mri_stability.R`): drought(11)/dry(29)/semi-arid(315)/
   wet(599)/monsoon(221 mm·yr⁻¹, bursts to 49 mm/day) all complete, match a tight reference to
   ≤3.5e-5, bounded soil sub-steps (6k–8k).
6. **Reverse-mode feasible** (Phase B, `mri_ad.hpp`, `mri_ad_test.R`): the tape of the scheme-as-run
   is the exact discrete adjoint — **48/48 gradients (4 scenarios × 12 traits) = frozen-schedule FD
   to 2e-9**; reverse **O(1) in |θ|**; reverse cost **O(M·n_macro), not O(M·n_micro)**. Near-bound
   apparent failures were a relative-error artifact on ~0 gradient components; smooth-floor removes
   the residual hard-clamp non-differentiability.

**Locked decisions:** MRI-GARK, solve-decoupled, explicit-slow, **component partition**; slow = big
block, fast = ≤5 soil states (static partition); coupling table = data (`MRIStepCoupling` shape);
inner = black-box adaptive RK behind an `InnerStepper` seam; macro grid = the daily forcing-kink
grid; **no new Layer-K nodes**. Rejected: global RODAS, MrGARK (fixed M), extrapolation/multistep/
QSS, the desingularizing chart (E2 measured it worse).

---

## B. Gap analysis — what the surrogate does NOT stress that TF24/TF24f require

The Phase-A/B surrogate coupling is deliberately clean: **scalar-aggregate, rank-1, u-independent**
coupling; **closed-form smooth** RHS; **fixed dimension**; well-conditioned. It validated the *scheme*
and the *tape mechanism*. It sidesteps (mapped to `engine-deepening-targets.md`):

| # | TF24/TF24f requires | surrogate has | threatens |
|---|---|---|---|
| 1 | **Nested leaf implicit solve in the fast RHS** (co-limitation `ci`, collar continuity root, optimum `q*` — 3 sign-definite scalar IFTs) producing per-layer uptake `E_i`; the aggregate `a(x,u)` is **expensive and u-dependent** (targets #1,#3) | one-line closed-form uptake; cheap u-independent `mean(θ)` | reverse-through-implicit inside the inner; the §6 aggregate-surrogate decision |
| 2 | **Divergent down-read** `ψ_soil=a_ψ(θ/θ_sat)^{−n}` in the differentiated path (target #3) | bounded `mean(θ)` | near-singular conditioning / adjoint amplification (F4); the real −0.91 collapse severity |
| 3 | **Hard events + recorded control flow**: `u_min` pinning (frozen active set), layer-crossing breakpoint, leaf shut-down early-exits classified `decide()` vs kink (targets #3,#4) | smoothstep + smooth floors, no events, no recorded decisions | the silent-gradient-bug class; event-as-active-IFT-root; `decide()`/replay firewall |
| 4 | **Growing dimension** (cohort birth; N grows monotonically; tape slots must not reallocate, `reserve_state`) | fixed M | two-level record/replay + MRI schedule under changing state size |
| 5 | **Real big-block coupling = the light-field scan** (rank-3 separable over N cohorts, resident-vs-mutant, `∂A/∂z`, near-diagonal band; crown-quadrature breakpoints) (target #6, #2) | trivial `mean(θ)` | the scan primitive, its L2 node record/replay, dg/dh |
| 6 | **TF24f tracked-`q` is a feedback loop** through the shared leaf solve (rate `k·G`; solved-vs-tracked changes the eigenvalue) (target #5) | decoupled leaky trackers | conditioning consequence of the feedback |
| 7 | **Severity**: real collapse 10³–10⁵ steps, corr −0.91 | ~5 micro-steps/macro | under-represents both the win and per-episode tape/checkpoint pressure |
| 8 | **Two regimes**: transient AND fixed-point (Eulerian BVP + IFT + eigenvalue) | transient only | MRI addresses only half of R7 |

Verdict-changing risks (test these before trusting the conclusion on real TF24): **#2 (divergent
read conditioning), #3 (events + recorded decisions), #1 (u-dependent expensive aggregate)**. The
rest (#4,#5,#6,#7) are build-plumbing that should compose.

---

## C. Is MRI a wholesale replacement of adaptive RK45? **No.**

- The **inner integrator on the 5 soil states is RK45**; the MRI **slow base method is also an ERK**.
  RK45 the *stepper* is retained and reused in both roles (RODAS is not needed — finding 1).
- MRI is **additive, opt-in** (a new driver beside the single-rate one, like `Method::rodas` beside
  `Method::rkck`). Existing K93/FF16 and single-rate TF24 runs are untouched.
- What it replaces, **for TF24 multirate runs only**, is the *single global adaptive advance loop*
  (`SolverInternal::advance_adaptive`, one step size for the whole patch) → a macro-step-over-the-
  slow-block + sub-cycle-the-soil loop that *calls* RK45 as a component.

---

## D. Forward work order — testing TF24/TF24f + MRI on challenging rainfall (no AD)

**The one real prerequisite: factor the patch coupling.** Today the patch integrates monolithically
(`Solver<patch_type>`, one `compute_rates()`, one step size — `scm.h`, `patch.h`). MRI needs the soil
sub-block RHS **callable in isolation given the slow state**, and vice-versa — the `StateView.u()`
surface (deepening-3): `f_u(θ, coupling_from_slow)` (needs the leaf solve given frozen/interpolated
cohorts over the leg) and `f_x(cohorts, aggregate_of_soil)`. This is the bulk of the work and is the
same interface reverse mode will reuse.

**Checklist:**
1. Partition declaration on the patch System: the ≤5 soil-state indices (fast) vs the rest (slow).
2. Factored coupling / `StateView.u()` — item above.
3. Leaf solve callable at soil-sub-step cadence — **exact-per-eval first** (correct; re-inflates cost
   during episodes — acceptable for a correctness+stability test; aggregate surrogate is the perf fix).
4. MRI macro driver wrapping RK45 (port `mri_core.hpp` to drive the real patch); **daily macro grid**.
5. TF24f: `q` (acclimation) is a **slow** state (rides the macro step); its rate reuses `G`.
6. Validation: MRI vs the **global RK45 run of the same patch** across challenging rainfall + the
   standing conservation invariants (Σ mass, offspring-production snapshots). Match + stable + cheaper.

**Cheaper bring-up rung (before the full refactor):** a **mode-flagged Lie-split wrapper** on the
patch — a System flag that zeroes cohort rates during the soil sub-cycle so the existing `Solver`
advances soil-only (cohorts frozen), then advances cohorts once per macro step. Validates the
concept + stability on the real coupling without the O(M) win.

**Not required (forward-only):** no AD/tape/rebind/record-replay-for-gradients, no smooth floors, no
implicit-node adjoint. Hard clamps are fine forward.

**Risk to report honestly:** exact-per-eval = O(N) leaf solves at every soil micro-step → partially
reinstates the amplifier during episodes; don't quote wall-time wins from it (quote step counts +
accuracy + stability). The aggregate surrogate (§6 of the plan) is the wall-time fix.

---

## E. Next steps (in order)

1. **[in progress] Higher-fidelity forward demonstration** — a surrogate carrying the TF24 coupling
   *structure* (nested collar-potential root reading divergent `ψ_soil`; hard `u_min` floor) integrated
   with the mode-flagged multirate mechanic, showing concept (matches global) + stability across
   drought→monsoon. This de-risks the real-patch bring-up without needing the `plant` build.
   (`plant` submodule is not checked out in the current environment.)
2. Check out + build `plant`; add the mode-flagged Lie-split bring-up to the real TF24/TF24f patch.
3. Factor the coupling (`StateView.u()`); port the MRI macro driver; validate across scenarios.
4. Only then: reverse-mode on the real coupling (the verdict-changing risks in §B).
