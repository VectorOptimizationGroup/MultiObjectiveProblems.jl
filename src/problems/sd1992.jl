"""
    SD()

Construct the fixed four-variable, two-objective `SD` problem.

The variables are bounded in `[1, 3] x [sqrt(2), 3]^2 x [1, 3]`, which keeps the
feasible set away from the poles of the second objective. An analytical Jacobian
is registered; objective Hessians are not registered.
"""
function SD()
    meta = META["SD"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(2) * x[1] + sqrt(T(2)) * (x[2] + x[3]) + x[4]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sqrt2 = sqrt(T(2))
        return T(2) / x[1] + T(2) * sqrt2 / x[2] + T(2) * sqrt2 / x[3] + T(2) / x[4]
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2)
        grad[2] = sqrt(T(2))
        grad[3] = sqrt(T(2))
        grad[4] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        sqrt2 = sqrt(T(2))
        grad[1] = -T(2) / x[1]^2
        grad[2] = -T(2) * sqrt2 / x[2]^2
        grad[3] = -T(2) * sqrt2 / x[3]^2
        grad[4] = -T(2) / x[4]^2
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = ([1.0, sqrt(2.0), sqrt(2.0), 1.0], [3.0, 3.0, 3.0, 3.0]),
        jacobian = (df1_dx, df2_dx),
    )
end