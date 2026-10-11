"""
    IM1()

Construct the fixed two-variable, two-objective `IM1` problem.

The first variable is bounded in `[1, 4]`, and the second in `[1, 2]`. An
analytical Jacobian is registered; objective Hessians are not registered.
"""
function IM1()
    meta = META["IM1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(2) * sqrt(x[1])
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1] * (one(T) - x[2]) + T(5)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T) / sqrt(x[1])
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T) - x[2]
        grad[2] = -x[1]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = ([1.0, 1.0], [4.0, 2.0]),
        jacobian = (df1_dx, df2_dx),
    )
end
