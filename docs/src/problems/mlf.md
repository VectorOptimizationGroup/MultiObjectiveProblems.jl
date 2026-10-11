# Molyneaux–Leyland–Favrat (MLF)

The `MLF1` and `MLF2` constructors implement two test problems introduced by
Molyneaux, Favrat, and Leyland [MLF2001](@cite). Huband et al. later cataloged
both problems as `MLF1` and `MLF2` [Huband2006](@cite).

!!! note "Corrected MLF1 formulation"
    Equation (2) of [MLF2001](@cite) prints the common factor as `1 * x/20`.
    Huband et al. identify `MLF1` as containing a typographical error and give
    the corrected factor ``1+x/20`` in Table XVI [Huband2006](@cite). `MLF1`
    implements this corrected formulation, which is also consistent with the
    objective amplitudes shown in Figures 3 and 4 of the original paper.

## Overview

Both constructors have fixed dimensions and register analytical Jacobians.
Objective Hessians are not registered. The catalog metadata classifies every
objective as not strictly convex (`:not_strictly_convex`).

| Problem | `nvar` | `nobj` | Registered bounds | Recommended working box |
|:---|---:|---:|:---|:---|
| `MLF1` | 1 | 2 | ``[0,20]`` | — |
| `MLF2` | 2 | 2 | None | ``[-100,100]^2`` |

`MLF2` is unconstrained: its recommended box is not registered in the returned
`MultiObjectiveProblem` and is not a domain stated in [MLF2001](@cite). It provides a
practical finite search region when an algorithm requires one.

## Mathematical formulations

### MLF1

Let ``F:[0,20]\to\mathbb{R}^2`` be defined by
``F(x)=(f_1(x),f_2(x))``. The corrected objectives are

```math
\begin{aligned}
f_1(x) &= \left(1+\frac{x_1}{20}\right)\sin(x_1),\\
f_2(x) &= \left(1+\frac{x_1}{20}\right)\cos(x_1).
\end{aligned}
```

### MLF2

Molyneaux, Favrat, and Leyland [MLF2001](@cite) formulate `MLF2` as a maximization problem. MultiObjectiveProblems.jl follows its minimization convention by
implementing the negative of each source objective. Thus, let
``F:\mathbb{R}^2\to\mathbb{R}^2`` be the implemented minimization vector
``F(x)=(f_1(x),f_2(x))``. Its components are

```math
\begin{aligned}
f_1(x) ={}& -5 + \frac{1}{200}\left[
\left(x_1^2+x_2-11\right)^2
+\left(x_1+x_2^2-7\right)^2\right],\\
f_2(x) ={}& -5 + \frac{1}{200}\left[
\left(4x_1^2+2x_2-11\right)^2
+\left(2x_1+4x_2^2-7\right)^2\right].
\end{aligned}
```

This sign change preserves the Pareto-optimal decision set, while reflecting
the Pareto front through the origin. Values in the source maximization
convention are obtained as `-eval_f(prob, x)`.

## Usage

```jldoctest mlf_usage
julia> using MultiObjectiveProblems

julia> using Random

julia> prob = MLF2();

julia> lower, upper = recommended_bounds(prob);

julia> rng = MersenneTwister(1234);

julia> α = rand(rng, prob.nvar);

julia> x = lower .+ α .* (upper .- lower);

julia> values = eval_f(prob, x);

julia> source_values = -values;

julia> J = eval_jacobian(prob, x);

julia> (length(values), size(J), length(source_values))
(2, (2, 2), 2)
```

## Constructor reference

```@docs
MultiObjectiveProblems.MLF1
MultiObjectiveProblems.MLF2
```
