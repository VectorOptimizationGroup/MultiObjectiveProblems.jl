"""
    SSFYY2()

Construct the fixed one-variable, two-objective `SSFYY2` problem.

The problem has no explicit variable bounds. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function SSFYY2()
    meta = META["SSFYY2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(10) + x[1]^2 - T(10) * cos(T(π) * x[1] / T(2))
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(4))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1] + T(5) * T(π) * sin(T(π) * x[1] / T(2))
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(4))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end
