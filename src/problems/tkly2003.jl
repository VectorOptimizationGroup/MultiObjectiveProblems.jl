"""
    TKLY1()

Construct the fixed four-variable, two-objective `TKLY1` problem.

The first variable is bounded in `[0.1, 1]` and the remaining ones in
`[0, 1]`. An analytical Jacobian is registered; objective Hessians are not
registered.
"""
function TKLY1()
    meta = META["TKLY1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    A = function (z::T) where {T <: AbstractFloat}
        u1 = (z - T(0.1)) / T(0.004)
        u2 = (z - T(0.9)) / T(0.4)
        return T(2) - exp(-u1^2) - T(0.8) * exp(-u2^2)
    end

    # Derivative of A; the factors 500 and 4 come from 2/0.004 and 2*0.8/0.4.
    dA = function (z::T) where {T <: AbstractFloat}
        u1 = (z - T(0.1)) / T(0.004)
        u2 = (z - T(0.9)) / T(0.4)
        return T(500) * exp(-u1^2) * u1 + T(4) * exp(-u2^2) * u2
    end

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return A(x[2]) * A(x[3]) * A(x[4]) / x[1]
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = zero(T)
        grad[3] = zero(T)
        grad[4] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a2 = A(x[2])
        a3 = A(x[3])
        a4 = A(x[4])
        numerator = a2 * a3 * a4
        grad[1] = -numerator / x[1]^2
        grad[2] = dA(x[2]) * a3 * a4 / x[1]
        grad[3] = a2 * dA(x[3]) * a4 / x[1]
        grad[4] = a2 * a3 * dA(x[4]) / x[1]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = ([0.1, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
        jacobian = (df1_dx, df2_dx),
    )
end
