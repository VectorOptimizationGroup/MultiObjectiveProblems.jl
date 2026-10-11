"""
    KW2()

Construct the fixed two-variable, two-objective `KW2` problem.

Kim and de Weck formulate Example 2 as maximizing `J(x)`; this constructor
minimizes `F(x) = -J(x)`. The Pareto-optimal decision set is preserved, while
reported objective values are sign-reversed. The variables are bounded in
`[-3, 3]^2`. An analytical Jacobian is registered; objective Hessians are not
registered.
"""
function KW2()
    meta = META["KW2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        term1 = -T(3) * (one(T) - x[1])^2 * exp(-x[1]^2 - (x[2] + one(T))^2)
        term2 = T(10) * (x[1] / T(5) - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2)
        term3 = T(3) * exp(-(x[1] + T(2))^2 - x[2]^2)
        term4 = -T(0.5) * (T(2) * x[1] + x[2])
        return term1 + term2 + term3 + term4
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        term1 = -T(3) * (one(T) + x[2])^2 * exp(-x[2]^2 - (one(T) - x[1])^2)
        term2 = T(10) * (-x[2] / T(5) + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2)
        term3 = T(3) * exp(-(T(2) - x[2])^2 - x[1]^2)
        return term1 + term2 + term3
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(6) * (one(T) - x[1]) * exp(-x[1]^2 - (x[2] + one(T))^2) +
                  T(6) * (one(T) - x[1])^2 * exp(-x[1]^2 - (x[2] + one(T))^2) * x[1] +
                  T(10) * (one(T) / T(5) - T(3) * x[1]^2) * exp(-x[1]^2 - x[2]^2) -
                  T(20) * (x[1] / T(5) - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2) * x[1] -
                  T(6) * exp(-(x[1] + T(2))^2 - x[2]^2) * (x[1] + T(2)) - one(T)

        grad[2] = T(6) * (one(T) - x[1])^2 * exp(-x[1]^2 - (x[2] + one(T))^2) * (x[2] + one(T)) -
                  T(50) * x[2]^4 * exp(-x[1]^2 - x[2]^2) -
                  T(10) * (x[1] / T(5) - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2) * T(2) * x[2] -
                  T(6) * exp(-(x[1] + T(2))^2 - x[2]^2) * x[2] - T(0.5)

        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -T(6) * (one(T) + x[2])^2 * exp(-x[2]^2 - (one(T) - x[1])^2) * (one(T) - x[1]) +
                  T(50) * x[1]^4 * exp(-x[1]^2 - x[2]^2) -
                  T(20) * (-x[2] / T(5) + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2) * x[1] -
                  T(6) * exp(-(T(2) - x[2])^2 - x[1]^2) * x[1]

        grad[2] = -T(6) * (one(T) + x[2]) * exp(-x[2]^2 - (one(T) - x[1])^2) +
                  T(6) * (one(T) + x[2])^2 * exp(-x[2]^2 - (one(T) - x[1])^2) * x[2] +
                  T(10) * (-one(T) / T(5) + T(3) * x[2]^2) * exp(-x[1]^2 - x[2]^2) -
                  T(20) * (-x[2] / T(5) + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2) * x[2] +
                  T(6) * exp(-(T(2) - x[2])^2 - x[1]^2) * (T(2) - x[2])

        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-3.0, n), fill(3.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end
