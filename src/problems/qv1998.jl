"""
    QV1(; nvar::Int = 16)

Construct the two-objective `QV1` problem.

Requires `nvar >= 1`. Each variable is bounded in `[-5.12, 5.12]`.
An analytical Jacobian is registered; objective Hessians are not registered.

The Jacobian rows are undefined at `x == zeros(nvar)` (row 1) and
`x == fill(1.5, nvar)` (row 2). Evaluating the corresponding row throws a
`DomainError`; objective values and the other row remain defined.
"""
function QV1(; nvar::Int = 16)
    n = nvar
    n >= 1 || throw(ArgumentError("nvar must be at least 1 for QV1"))
    meta = META["QV1"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        twoπ = T(2) * T(π)
        s = zero(T)
        @inbounds for i in 1:n
            s += x[i]^2 - T(10) * cos(twoπ * x[i]) + T(10)
        end
        return (s / T(n))^T(0.25)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        twoπ = T(2) * T(π)
        s = zero(T)
        @inbounds for i in 1:n
            y = x[i] - T(1.5)
            s += y^2 - T(10) * cos(twoπ * y) + T(10)
        end
        return (s / T(n))^T(0.25)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        all(iszero, x) && throw(DomainError(
            Tuple(x),
            "QV1 Jacobian row 1 is undefined at the first objective minimizer " *
            "(0, ..., 0).",
        ))

        twoπ = T(2) * T(π)
        s = zero(T)
        @inbounds for i in 1:n
            s += x[i]^2 - T(10) * cos(twoπ * x[i]) + T(10)
        end

        factor = T(0.25) * (s / T(n))^(-T(0.75)) / T(n)
        @inbounds for i in 1:n
            grad[i] = factor * (T(2) * x[i] + T(20) * T(π) * sin(twoπ * x[i]))
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        all(==(T(1.5)), x) && throw(DomainError(
            Tuple(x),
            "QV1 Jacobian row 2 is undefined at the second objective minimizer " *
            "(1.5, ..., 1.5).",
        ))

        twoπ = T(2) * T(π)
        s = zero(T)
        @inbounds for i in 1:n
            y = x[i] - T(1.5)
            s += y^2 - T(10) * cos(twoπ * y) + T(10)
        end

        factor = T(0.25) * (s / T(n))^(-T(0.75)) / T(n)
        @inbounds for i in 1:n
            y = x[i] - T(1.5)
            grad[i] = factor * (T(2) * y + T(20) * T(π) * sin(twoπ * y))
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-5.12, n), fill(5.12, n)),
        jacobian = (df1_dx, df2_dx),
    )
end
