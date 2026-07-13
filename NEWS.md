## Odelia 0.0.0.9000

* New `odelia::ad::directional_derivative` (`directional_derivative.hpp`): a forward-over-reverse directional derivative that stays active on an outer reverse (adjoint) tape (#38). It computes `df/dx` along one seeded direction exactly — the analytic derivative of the code, with clamps/kinks handled correctly — so an enclosing reverse sweep differentiates it further into a mixed second derivative with no finite-difference step. The working type composes as `FReal<AReal<double>>` (tangent-over-adjoint, #35). Intended for terms like plant's density-transport `dg/dh` evaluated inside a parameter reverse sweep (plant#39).

* `basic_interpolator` active-query reads now offer an explicit choice of query-point derivative (#38): `eval(Q)` / `operator()(Q)` carry the analytic tangent `d(value)/d(query)`; the new `eval_frozen_query(Q)` freezes it (reads at `xad::value(query)`) while still carrying the knot-value derivatives. The analytic tangent of an under-resolved spline is unreliable for an evolving ODE-state query point and compounds across a time integration (the bug behind plant#39); callers on a rate path should use `eval_frozen_query`.

Odelia is a new package, arsising out of https://github.com/traitecoevo/plant/. In that project, Rich FitzJohn built a custom ODE solver, using Runge-Kutta4-5 method, in C++. I'm spinning that code out into a package, as I want to use it elsewhere. 

* New implicit, adaptive-step **RODAS4(3)** Rosenbrock stepper for stiff systems, selectable via `method = "rodas"` when constructing a solver (#35). It reuses the existing adaptive step-size controller and obtains an exact Jacobian by forward-mode automatic differentiation; systems opt in by providing a `template<class U> rebind<U> rebind_from()` method (the same double->AD lift the gradient driver uses). The explicit RKCK 4(5) method (`method = "rkck"`) remains the default.

* `odelia` now loads its shared library with global symbol visibility in `.onLoad`, so packages that `LinkingTo: odelia` and instantiate `Solver` can resolve the compiled XAD runtime symbols at load time without per-package linker hacks (#26).

