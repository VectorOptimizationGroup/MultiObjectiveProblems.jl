"""
    AbstractDimensionSpec

Abstract supertype for catalog dimension specifications.
"""
abstract type AbstractDimensionSpec end

"""
    FixedDimension(nvar, nobj)

Dimension specification with fixed numbers of variables and objectives.
"""
struct FixedDimension <: AbstractDimensionSpec
    nvar::Int
    nobj::Int
    function FixedDimension(nvar::Integer, nobj::Integer)
        return new(Int(nvar), Int(nobj))
    end
end

"""
    VariableNvar(default_nvar, nobj)

Dimension specification with configurable `nvar`, defaulting to `default_nvar`,
and fixed `nobj`.
"""
struct VariableNvar <: AbstractDimensionSpec
    default_nvar::Int
    nobj::Int
    function VariableNvar(default_nvar::Integer, nobj::Integer)
        return new(Int(default_nvar), Int(nobj))
    end
end

"""
    VariableNobj(nvar, default_nobj)

Dimension specification with fixed `nvar` and configurable `nobj`, defaulting
to `default_nobj`.
"""
struct VariableNobj <: AbstractDimensionSpec
    nvar::Int
    default_nobj::Int
    function VariableNobj(nvar::Integer, default_nobj::Integer)
        return new(Int(nvar), Int(default_nobj))
    end
end

"""
    IndependentDimension(default_nvar, default_nobj)

Dimension specification with independently configurable `nvar` and `nobj`,
using the supplied defaults.
"""
struct IndependentDimension <: AbstractDimensionSpec
    default_nvar::Int
    default_nobj::Int
    function IndependentDimension(default_nvar::Integer, default_nobj::Integer)
        return new(Int(default_nvar), Int(default_nobj))
    end
end

"""
    ParametricDimension(default_k, default_nobj)

Dimension specification with configurable `k` and `nobj`, using the supplied
defaults and satisfying `nvar = k + nobj - 1`.
"""
struct ParametricDimension <: AbstractDimensionSpec
    default_k::Int
    default_nobj::Int
    function ParametricDimension(default_k::Integer, default_nobj::Integer)
        return new(Int(default_k), Int(default_nobj))
    end
end

"""
    CoupledDimension(default_nvar, default_nobj)

Dimension specification with configurable `nvar` and coupled `nobj`, using the
supplied defaults. Configurations preserve
`nvar - nobj = default_nvar - default_nobj`.
"""
struct CoupledDimension <: AbstractDimensionSpec
    default_nvar::Int
    default_nobj::Int
    function CoupledDimension(default_nvar::Integer, default_nobj::Integer)
        return new(Int(default_nvar), Int(default_nobj))
    end
end

_default_nvar(spec::FixedDimension) = spec.nvar
_default_nobj(spec::FixedDimension) = spec.nobj
_default_nvar(spec::VariableNvar) = spec.default_nvar
_default_nobj(spec::VariableNvar) = spec.nobj
_default_nvar(spec::VariableNobj) = spec.nvar
_default_nobj(spec::VariableNobj) = spec.default_nobj
_default_nvar(spec::IndependentDimension) = spec.default_nvar
_default_nobj(spec::IndependentDimension) = spec.default_nobj
_default_nvar(spec::ParametricDimension) = spec.default_k + spec.default_nobj - 1
_default_nobj(spec::ParametricDimension) = spec.default_nobj
_default_nvar(spec::CoupledDimension) = spec.default_nvar
_default_nobj(spec::CoupledDimension) = spec.default_nobj

"""
    ProblemMeta

Static metadata for a benchmark problem in the package catalog. `dimension`
records the default dimensions and how they can vary. `ncon_eq` and `ncon_ineq`
count general constraints, excluding variable bounds. Derivative flags record
registered analytical evaluators separately for objectives and constraints.
`strict_convexity` contains `:strictly_convex` or `:not_strictly_convex` for
each objective of the default instance, or `nothing` when unavailable.
"""
struct ProblemMeta
    dimension::AbstractDimensionSpec
    name::String
    has_bounds::Bool
    has_jacobian::Bool
    has_hessian::Bool
    ncon_eq::Int
    ncon_ineq::Int
    has_constraint_jacobian::Bool
    has_constraint_hessian::Bool
    strict_convexity::Union{Nothing, Vector{Symbol}}
    function ProblemMeta(;
        dimension::AbstractDimensionSpec,
        name::AbstractString,
        has_bounds::Bool = false,
        has_jacobian::Bool = false,
        has_hessian::Bool = false,
        ncon_eq::Integer = 0,
        ncon_ineq::Integer = 0,
        has_constraint_jacobian::Bool = false,
        has_constraint_hessian::Bool = false,
        strict_convexity::Union{Nothing, Vector{Symbol}} = nothing
    )
        return new(
            dimension,
            String(name),
            has_bounds,
            has_jacobian,
            has_hessian,
            Int(ncon_eq),
            Int(ncon_ineq),
            has_constraint_jacobian,
            has_constraint_hessian,
            strict_convexity,
        )
    end
end

"""
    default_nvar(meta::ProblemMeta) -> Int

Return the number of variables in the metadata's default instance.
"""
default_nvar(meta::ProblemMeta) = _default_nvar(meta.dimension)::Int

"""
    default_nobj(meta::ProblemMeta) -> Int

Return the number of objectives in the metadata's default instance.
"""
default_nobj(meta::ProblemMeta) = _default_nobj(meta.dimension)::Int

"""
    MultiObjectiveProblem

Concrete evaluable instance of a benchmark problem.

Benchmark constructors such as `ZDT1()` and `AP1()` return `MultiObjectiveProblem`
instances. The fields `nvar`, `nobj`, and `ncon` give the numbers of variables,
objectives, and general constraints. `bounds` is either `(lower, upper)` or
`nothing`. Objective and constraint evaluators are stored in `f` and `c`, and
derivative fields are `nothing` when no analytical evaluator is registered.

General constraints follow `lcon <= c(x) <= ucon`; equalities are the rows for
which the corresponding lower and upper bounds are equal. Static catalog
information belongs to `ProblemMeta`.
"""
struct MultiObjectiveProblem{F, J, H, B, C, CJ, CH, LC, UC}
    name::String
    nvar::Int
    nobj::Int
    f::F
    jacobian::J
    hessian::H
    bounds::B
    ncon::Int
    c::C
    constraint_jacobian::CJ
    constraint_hessian::CH
    lcon::LC
    ucon::UC
end

function MultiObjectiveProblem(
    nvar::Integer,
    nobj::Integer,
    f;
    name::AbstractString = "Unnamed problem",
    bounds = nothing,
    jacobian = nothing,
    hessian = nothing,
    c = (),
    lcon = (),
    ucon = (),
    constraint_jacobian = nothing,
    constraint_hessian = nothing
)
    nvar = Int(nvar)
    nobj = Int(nobj)
    ncon = length(c)

    return MultiObjectiveProblem(
        String(name),
        nvar,
        nobj,
        f,
        jacobian,
        hessian,
        bounds,
        ncon,
        c,
        constraint_jacobian,
        constraint_hessian,
        lcon,
        ucon,
    )
end
