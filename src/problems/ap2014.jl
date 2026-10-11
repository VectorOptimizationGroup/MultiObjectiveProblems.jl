"""
    AP1()

Construct the fixed two-variable, three-objective `AP1` problem.

The variables are bounded in `[-10, 10]^2`. An analytical Jacobian and objective
Hessians are registered.
"""
function AP1()
    meta = META["AP1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(0.25) * ((x[1] - one(T))^4 + T(2.0) * (x[2] - T(2.0))^4)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return exp((x[1] + x[2]) / T(2.0)) + x[1]^2 + x[2]^2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) / T(6.0)) * (exp(-x[1]) + T(2.0) * exp(-x[2]))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = (x[1] - one(T))^3
        grad[2] = T(2.0) * (x[2] - T(2.0))^3
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp_term = exp((x[1] + x[2]) / T(2.0))
        grad[1] = T(0.5) * exp_term + T(2.0) * x[1]
        grad[2] = T(0.5) * exp_term + T(2.0) * x[2]
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -(one(T) / T(6.0)) * exp(-x[1])
        grad[2] = -(one(T) / T(3.0)) * exp(-x[2])
        return grad
    end

    h1 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = T(3.0) * (x[1] - one(T))^2
        H[2, 2] = T(6.0) * (x[2] - T(2.0))^2
        return H
    end

    h2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp_term = exp((x[1] + x[2]) / T(2.0))
        c = T(0.25) * exp_term
        H[1, 1] = c + T(2.0)
        H[2, 2] = c + T(2.0)
        H[1, 2] = c
        H[2, 1] = c
        return H
    end

    h3 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = one(T) / T(6.0) * exp(-x[1])
        H[2, 2] = one(T) / T(3.0) * exp(-x[2])
        return H
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-10.0, n), fill(10.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
        hessian = (h1, h2, h3),
    )
end

"""
    AP2()

Construct the fixed one-variable, two-objective `AP2` problem.

The variable is bounded in `[-100, 100]`. An analytical Jacobian and objective
Hessians are registered.
"""
function AP2()
    meta = META["AP2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 - T(4.0)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - one(T))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2.0) * x[1]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2.0) * (x[1] - one(T))
        return grad
    end

    h1 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = T(2.0)
        return H
    end

    h2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = T(2.0)
        return H
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-100.0, n), fill(100.0, n)),
        jacobian = (df1_dx, df2_dx),
        hessian = (h1, h2),
    )
end

"""
    AP3()

Construct the fixed two-variable, two-objective `AP3` problem.

The variables are bounded in `[-100, 100]^2`. An analytical Jacobian and
objective Hessians are registered.
"""
function AP3()
    meta = META["AP3"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(0.25) * ((x[1] - one(T))^4 + T(2.0) * (x[2] - T(2.0))^4)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[2] - x[1]^2)^2 + (one(T) - x[1])^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = (x[1] - one(T))^3
        grad[2] = T(2.0) * (x[2] - T(2.0))^3
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -T(4.0) * x[1] * (x[2] - x[1]^2) - T(2.0) * (one(T) - x[1])
        grad[2] = T(2.0) * (x[2] - x[1]^2)
        return grad
    end

    h1 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = T(3.0) * (x[1] - one(T))^2
        H[2, 2] = T(6.0) * (x[2] - T(2.0))^2
        return H
    end

    h2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = -T(4.0) * (x[2] - x[1]^2) + T(8.0) * x[1]^2 + T(2.0)
        H[1, 2] = -T(4.0) * x[1]
        H[2, 1] = H[1, 2]
        H[2, 2] = T(2.0)
        return H
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-100.0, n), fill(100.0, n)),
        jacobian = (df1_dx, df2_dx),
        hessian = (h1, h2),
    )
end

"""
    AP4()

Construct the fixed three-variable, three-objective `AP4` problem.

The variables are bounded in `[-10, 10]^3`. An analytical Jacobian and objective
Hessians are registered.
"""
function AP4()
    meta = META["AP4"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) / T(9.0)) * (
            (x[1] - one(T))^4 +
            T(2.0) * (x[2] - T(2.0))^4 +
            T(3.0) * (x[3] - T(3.0))^4
        )
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return exp((x[1] + x[2] + x[3]) / T(3.0)) + x[1]^2 + x[2]^2 + x[3]^2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) / T(12.0)) * (T(3.0) * exp(-x[1]) + T(4.0) * exp(-x[2]) + T(3.0) * exp(-x[3]))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = (T(4.0) / T(9.0)) * (x[1] - one(T))^3
        grad[2] = (T(8.0) / T(9.0)) * (x[2] - T(2.0))^3
        grad[3] = (T(12.0) / T(9.0)) * (x[3] - T(3.0))^3
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp_term = exp((x[1] + x[2] + x[3]) / T(3.0))
        grad[1] = (one(T) / T(3.0)) * exp_term + T(2.0) * x[1]
        grad[2] = (one(T) / T(3.0)) * exp_term + T(2.0) * x[2]
        grad[3] = (one(T) / T(3.0)) * exp_term + T(2.0) * x[3]
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -(one(T) / T(4.0)) * exp(-x[1])
        grad[2] = -(one(T) / T(3.0)) * exp(-x[2])
        grad[3] = -(one(T) / T(4.0)) * exp(-x[3])
        return grad
    end

    h1 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = (T(12.0) / T(9.0)) * (x[1] - one(T))^2
        H[2, 2] = (T(24.0) / T(9.0)) * (x[2] - T(2.0))^2
        H[3, 3] = (T(36.0) / T(9.0)) * (x[3] - T(3.0))^2
        return H
    end

    h2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t = (one(T) / T(9.0)) * exp((x[1] + x[2] + x[3]) / T(3.0))
        for i in 1:n, j in 1:n
            H[i, j] = t
        end
        for i in 1:n
            H[i, i] += T(2.0)
        end
        return H
    end

    h3 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[1, 1] = (one(T) / T(4.0)) * exp(-x[1])
        H[2, 2] = (one(T) / T(3.0)) * exp(-x[2])
        H[3, 3] = (one(T) / T(4.0)) * exp(-x[3])
        return H
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-10.0, n), fill(10.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
        hessian = (h1, h2, h3),
    )
end