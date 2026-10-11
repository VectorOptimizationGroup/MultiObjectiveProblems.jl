# Preuss–Naujoks–Rudolph (PNR)

This family is represented by the `PNR` constructor. It implements Case 1 of
the `TWO-ON-ONE` test problem introduced in “Pareto Set and EMOA Behavior for
Simple Multimodal Multiobjective Functions” [PNR2006](@cite).

!!! note "Source specialization and recommended working box"
    The source defines the parameterized `TWO-ON-ONE` mapping on
    ``\mathbb{R}^2``. `PNR` specializes it to the Case 1 parameters
    ``c=10`` and ``d=k=l=0``. The box ``[-2,2]^2`` is recommended for finite
    searches, but it is neither registered in the returned `MultiObjectiveProblem` nor
    stated as the domain of the source formulation.

## Overview

`PNR` has fixed dimensions and is unconstrained.

| Problem | Source case | `nvar` | `nobj` | Registered bounds | Recommended working box |
|:---|---:|---:|---:|:---|:---|
| `PNR` | 1 | 2 | 2 | None | ``[-2,2]^2`` |

The recommended box provides a practical finite search region when an
algorithm requires one. An analytical Jacobian is registered. Hessians are
not registered. The catalog metadata classifies the first objective as not
strictly convex (`:not_strictly_convex`) and the second as strictly convex
(`:strictly_convex`).

## Mathematical formulation

The source parameterization is

```math
\begin{aligned}
f_1(x) &= x_1^4+x_2^4-x_1^2+x_2^2-cx_1x_2+dx_1+20,\\
f_2(x) &= (x_1-k)^2+(x_2-l)^2.
\end{aligned}
```

Case 1 sets ``c=10`` and ``d=k=l=0``. Therefore, the constructor implements
``F:\mathbb{R}^2\to\mathbb{R}^2``, with ``F(x)=(f_1(x),f_2(x))`` and

```math
\begin{aligned}
f_1(x) &= x_1^4+x_2^4-x_1^2+x_2^2-10x_1x_2+20,\\
f_2(x) &= x_1^2+x_2^2.
\end{aligned}
```

## Usage

```jldoctest pnr_usage
julia> using MultiObjectiveProblems

julia> using Random

julia> prob = PNR();

julia> lower, upper = recommended_bounds(prob);

julia> rng = MersenneTwister(1234);

julia> α = rand(rng, prob.nvar);

julia> x = lower .+ α .* (upper .- lower);

julia> values = eval_f(prob, x);

julia> J = eval_jacobian(prob, x);

julia> (length(values), size(J))
(2, (2, 2))
```

## Constructor reference

```@docs
MultiObjectiveProblems.PNR
```
