"""
    FDS(; nvar::Int = 5)

Construct the three-objective `FDS` problem.

Requires `nvar >= 1`. Each variable is bounded in `[-2, 2]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function FDS(; nvar::Int = 5)
    n = nvar
    n >= 1 || throw(ArgumentError("nvar must be at least 1 for FDS"))
    meta = META["FDS"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_val = zero(T)
        for i in 1:n
            sum_val += T(i) * (x[i] - T(i))^4
        end
        return sum_val / (T(n)^2)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_x = zero(T)
        norm2 = zero(T)
        for i in 1:n
            sum_x += x[i]
            norm2 += x[i]^2
        end
        return exp(sum_x / T(n)) + norm2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_val = zero(T)
        for i in 1:n
            sum_val += T(i) * T(n - i + 1) * exp(-x[i])
        end
        return sum_val / (T(n) * T(n + 1))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in 1:n
            grad[i] = T(4) * T(i) * (x[i] - T(i))^3 / (T(n)^2)
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_x = zero(T)
        for i in 1:n
            sum_x += x[i]
        end
        exp_term = exp(sum_x / T(n))
        for i in 1:n
            grad[i] = exp_term / T(n) + T(2) * x[i]
        end
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in 1:n
            grad[i] = -T(i) * T(n - i + 1) * exp(-x[i]) / (T(n) * T(n + 1))
        end
        return grad
    end

    bounds = (fill(-2.0, n), fill(2.0, n))

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = bounds,
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
