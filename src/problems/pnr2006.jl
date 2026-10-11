"""
    PNR()

Construct the fixed two-variable, two-objective `PNR` problem.

The problem has no explicit variable bounds. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function PNR()
    meta = META["PNR"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^4 + x[2]^4 - x[1]^2 + x[2]^2 - T(10) * x[1] * x[2] + T(20)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + x[2]^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(4) * x[1]^3 - T(2) * x[1] - T(10) * x[2]
        grad[2] = T(4) * x[2]^3 + T(2) * x[2] - T(10) * x[1]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = T(2) * x[2]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end
