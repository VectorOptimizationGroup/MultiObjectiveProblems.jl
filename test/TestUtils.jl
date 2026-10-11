module TestUtils

using Test
using Random
using LinearAlgebra
using FiniteDiff
using MultiObjectiveProblems

export instantiate_with_dimension, sample_x, check_jacobian, check_hessian, dims

const FAST = get(ENV, "MO_FAST", "1") == "1"
dims_fast() = (5, 10)
dims_full() = (5, 10, 20, 30, 50)
dims() = FAST ? dims_fast() : dims_full()

# `ATOL` is the floor of the relative-error denominator, needed because some
# Jacobians are numerically zero over their box. `RTOL` is set against the
# measured spread of the catalog, whose analytical derivatives agree with the
# extended-precision reference to 3.5e-15 at the 99th percentile and 5.1e-14 at
# worst. That worst case is `MOP5`, where the same formula evaluated in
# `BigFloat` agrees to 6.2e-47: the residue is the Float64 rounding of
# `sin(x₁²+x₂²)` over a box that carries the argument to ~843, and it grows
# with the argument, so the tolerance keeps room for families with wider boxes.
const ATOL = 1e-8
const RTOL = 1e-10

relok(A, B; atol=ATOL, rtol=RTOL) = (norm(A - B) / max(norm(B), atol)) <= rtol

# Uniform point in the problem's bounds or, for unbounded problems, in the box
# from `recommended_bounds`.
function sample_x(prob::MultiObjectiveProblem; rng=Random.MersenneTwister(42))
    l, u = recommended_bounds(prob)
    return [Float64(l[i]) + rand(rng) * (Float64(u[i]) - Float64(l[i])) for i in 1:prob.nvar]
end

# The analytical Jacobian is evaluated in `Float64`, the way callers use it, and
# compared against central differences taken in extended precision. A `Float64`
# difference is not accurate enough to be the reference: over the registered
# boxes its own error against the analytical value reaches 3e-4, past the
# tolerance applied here, while the `BigFloat` difference stays below 1e-13 for
# every problem in the catalog. The package is generic over the float type, so
# the problem closures take `BigFloat` unchanged and FiniteDiff sizes its own
# step from `eps(BigFloat)`.
function check_jacobian(f, J, x; atol=ATOL, rtol=RTOL, precision=256)
    Jfd = setprecision(BigFloat, precision) do
        FiniteDiff.finite_difference_jacobian(f, BigFloat.(x), Val(:central))
    end
    Jx = J(x)
    ok = relok(Jx, Jfd; atol=atol, rtol=rtol)
    return ok, Float64(norm(Jx - Jfd) / max(norm(Jfd), atol))
end

# Same reference as `check_jacobian`, for one objective's Hessian. In `Float64`
# the difference reaches 1.4e-7 against the analytical value, within an order of
# magnitude of the tolerance; in extended precision it stays below 1e-16.
function check_hessian(fi, Hx, x; atol=ATOL, rtol=RTOL, precision=256)
    Hfd = setprecision(BigFloat, precision) do
        FiniteDiff.finite_difference_hessian(fi, BigFloat.(x), Val(:hcentral))
    end
    ok = relok(Hx, Hfd; atol=atol, rtol=rtol)
    return ok, Float64(norm(Hx - Hfd) / max(norm(Hfd), atol))
end

function instantiate_with_dimension(name::String, n::Int)
    constructor = getfield(MultiObjectiveProblems, Symbol(name))
    return instantiate_with_dimension(constructor, n, MultiObjectiveProblems.META[name].dimension)
end

instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.FixedDimension) = constructor()

instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.VariableNvar) = constructor(nvar=n)

instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.VariableNobj) = constructor(nobj=n)

instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.IndependentDimension) =
    constructor(nvar=n, nobj=max(2, n - 1))

function instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.ParametricDimension)
    m = 3
    k = max(1, n - m + 1)
    return constructor(k=k, nobj=m)
end

instantiate_with_dimension(constructor, n, ::MultiObjectiveProblems.CoupledDimension) =
    constructor(nvar=n)

end # module
