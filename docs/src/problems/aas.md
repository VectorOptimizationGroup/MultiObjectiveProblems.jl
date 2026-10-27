# Amaral–Assunção–Souza (AAS)

This family comprises the `AAS1` and `AAS2` constructors. Both problems are
drawn from “A Partially Derivative-Free Proximal Method for Composite
Multiobjective Optimization in the Hölder Setting” [AAS2025](@cite).

## Overview

Both constructors have `nvar = 2` and `nobj = 2`. Their bounds differ as shown
below.

| Problem | `nvar` | `nobj` | Lower bound | Upper bound |
|:---|---:|---:|---:|---:|
| `AAS1` | 2 | 2 | -2.0 | 2.0 |
| `AAS2` | 2 | 2 | -5.0 | 5.0 |

The bounds in each row apply componentwise to both variables. Analytical Jacobians are registered for both constructors. **Hessians are not registered**. Strict-convexity information is not available (`nothing`) for either problem.

!!! note "Gradients where a power term vanishes"
    Each ``\frac{\mu}{p}\lVert y\rVert_p^p`` objective, with ``y`` affine in
    ``x`` and ``p > 1``, has gradient ``\mu\, M^\top\big(\operatorname{sign}(y)
    \odot \lvert y\rvert^{p-1}\big)``, where ``M`` is ``D``, ``D_1`` or
    ``D_2``. It is defined where a component ``y_i`` vanishes, and that
    component contributes zero. Because ``p - 1`` is between ``0.002`` and
    ``0.003``, ``\lvert y_i\rvert^{p-1}`` is still about ``0.95`` at
    ``\lvert y_i\rvert = 10^{-8}``: the gradient varies sharply near those
    points, and an evaluation in `Float32` can round ``y_i`` to zero where the
    exact value is not.

## Mathematical formulations

The formulas below describe the objective functions implemented by the
constructors. In both cases, let ``F:\mathbb{R}^2 \to \mathbb{R}^2`` be
defined by ``F(x)=(f_1(x),f_2(x))``, where
``x = (x_1,x_2) \in \mathbb{R}^2``.

### AAS1

The objectives are

```math
\begin{aligned}
f_1(x) &= \frac{1}{2}\lVert Ax-b\rVert_2^2, &
f_2(x) &= \frac{\mu}{p}\lVert Dx\rVert_p^p.
\end{aligned}
```

The implemented parameters are

```math
A = \begin{bmatrix}2.0 & 0.5\\0.5 & 1.5\end{bmatrix},
\qquad
b = \begin{bmatrix}1.0\\-0.5\end{bmatrix},
\qquad
p = 1.003,
\qquad
\mu = 0.9,
\qquad
D = \begin{bmatrix}1.0 & 0.8\\0.3 & 1.2\end{bmatrix}.
```

### AAS2

The objectives are

```math
\begin{aligned}
f_1(x) &= \frac{\mu_1}{p_1}\lVert D_1(x-c_1)\rVert_{p_1}^{p_1}, &
f_2(x) &= \frac{\mu_2}{p_2}\lVert D_2(x-c_2)\rVert_{p_2}^{p_2}.
\end{aligned}
```

The implemented parameters are

```math
\begin{aligned}
p_1 &= 1.003, &
\mu_1 &= 1.2, &
D_1 &= \begin{bmatrix}1.2 & -0.3\\0.4 & 1.5\end{bmatrix}, &
c_1 &= \begin{bmatrix}1.5\\-1.0\end{bmatrix},\\[6pt]
p_2 &= 1.002, &
\mu_2 &= 0.8, &
D_2 &= \begin{bmatrix}1.8 & 0.5\\-0.2 & 1.1\end{bmatrix}, &
c_2 &= \begin{bmatrix}-1.2\\0.8\end{bmatrix}.
\end{aligned}
```

## Usage

```jldoctest aas_usage
julia> using MultiObjectiveProblems

julia> using Random

julia> prob = AAS1();

julia> lower, upper = recommended_bounds(prob);

julia> rng = MersenneTwister(1234);

julia> α = rand(rng, prob.nvar);

julia> x = lower .+ α .* (upper .- lower);

julia> values = eval_f(prob, x);

julia> length(values)
2
```

## Constructor reference

```@docs
MultiObjectiveProblems.AAS1
MultiObjectiveProblems.AAS2
```
