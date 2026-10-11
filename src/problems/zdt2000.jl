"""
    ZDT1(; nvar::Int = 30)

Construct the two-objective `ZDT1` problem.

Requires `nvar >= 2`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.

At `x[1] == 0`, the second Jacobian row is undefined and its evaluation
throws a `DomainError`. Objective values and the first row remain defined.
"""
function ZDT1(; nvar::Int = 30)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for ZDT1"))
    meta = META["ZDT1"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    g = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for i in 2:n
            s += x[i]
        end
        return one(T) + T(9) * s / T(n - 1)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        gx = g(x)
        return gx * (one(T) - sqrt(x[1] / gx))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        x[1] > zero(T) || throw(DomainError(
            x[1],
            "ZDT1 Jacobian row 2 requires x₁ > 0; " *
            "its x₁ derivative is undefined at x₁ = 0.",
        ))

        gx = g(x)
        sqrt_ratio = sqrt(x[1] / gx)
        grad[1] = -T(0.5) / sqrt_ratio
        dg_dxi = T(9) / T(n - 1)
        @inbounds for i in 2:n
            grad[i] = dg_dxi * (one(T) - T(0.5) * sqrt_ratio)
        end
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    ZDT2(; nvar::Int = 30)

Construct the two-objective `ZDT2` problem.

Requires `nvar >= 2`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function ZDT2(; nvar::Int = 30)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for ZDT2"))
    meta = META["ZDT2"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    g = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for i in 2:n
            s += x[i]
        end
        return one(T) + T(9) * s / T(n - 1)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        gx = g(x)
        return gx * (one(T) - (x[1] / gx)^2)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        gx = g(x)
        ratio = x[1] / gx
        grad[1] = -T(2) * ratio
        dg_dxi = T(9) / T(n - 1)
        @inbounds for i in 2:n
            grad[i] = dg_dxi * (one(T) + ratio^2)
        end
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    ZDT3(; nvar::Int = 30)

Construct the two-objective `ZDT3` problem.

Requires `nvar >= 2`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.

At `x[1] == 0`, the second Jacobian row is undefined and its evaluation
throws a `DomainError`. Objective values and the first row remain defined.
"""
function ZDT3(; nvar::Int = 30)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for ZDT3"))
    meta = META["ZDT3"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    g = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for i in 2:n
            s += x[i]
        end
        return one(T) + T(9) * s / T(n - 1)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        gx = g(x)
        ratio = x[1] / gx
        return gx * (one(T) - sqrt(ratio) - ratio * sin(T(10) * T(π) * x[1]))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        x[1] > zero(T) || throw(DomainError(
            x[1],
            "ZDT3 Jacobian row 2 requires x₁ > 0; " *
            "its x₁ derivative is undefined at x₁ = 0.",
        ))

        gx = g(x)
        ratio = x[1] / gx
        sqrt_ratio = sqrt(ratio)
        angle = T(10) * T(π) * x[1]
        grad[1] = -T(0.5) / sqrt_ratio - sin(angle) - T(10) * T(π) * x[1] * cos(angle)
        dg_dxi = T(9) / T(n - 1)
        @inbounds for i in 2:n
            grad[i] = dg_dxi * (one(T) - T(0.5) * sqrt_ratio)
        end
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    ZDT4(; nvar::Int = 10)

Construct the two-objective `ZDT4` problem.

Requires `nvar >= 2`. The first variable is bounded in `[0, 1]`;
the remaining variables are bounded in `[-5, 5]`.
An analytical Jacobian is registered; objective Hessians are not registered.

At `x[1] == 0`, the second Jacobian row is undefined and its evaluation
throws a `DomainError`. Objective values and the first row remain defined.
"""
function ZDT4(; nvar::Int = 10)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for ZDT4"))
    meta = META["ZDT4"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    g = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for i in 2:n
            s += x[i]^2 - T(10) * cos(T(4) * T(π) * x[i])
        end
        return one(T) + T(10) * T(n - 1) + s
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        gx = g(x)
        return gx * (one(T) - sqrt(x[1] / gx))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        x[1] > zero(T) || throw(DomainError(
            x[1],
            "ZDT4 Jacobian row 2 requires x₁ > 0; " *
            "its x₁ derivative is undefined at x₁ = 0.",
        ))

        gx = g(x)
        sqrt_ratio = sqrt(x[1] / gx)
        grad[1] = -T(0.5) / sqrt_ratio
        @inbounds for i in 2:n
            dg_dxi = T(2) * x[i] + T(40) * T(π) * sin(T(4) * T(π) * x[i])
            grad[i] = dg_dxi * (one(T) - T(0.5) * sqrt_ratio)
        end
        return grad
    end

    lower = fill(-5.0, n)
    upper = fill(5.0, n)
    lower[1] = 0.0
    upper[1] = 1.0

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (lower, upper),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    ZDT6(; nvar::Int = 10)

Construct the two-objective `ZDT6` problem.

Requires `nvar >= 2`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.

When all variables except the first are zero, the second Jacobian row is
undefined and its evaluation throws a `DomainError`. Objective values and
the first row remain defined.
"""
function ZDT6(; nvar::Int = 10)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for ZDT6"))
    meta = META["ZDT6"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return one(T) - exp(-T(4) * x[1]) * sin(T(6) * T(π) * x[1])^6
    end

    g = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for i in 2:n
            s += x[i]
        end
        return one(T) + T(9) * (s / T(n - 1))^T(0.25)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        f1x = f1(x)
        gx = g(x)
        return gx * (one(T) - (f1x / gx)^2)
    end

    df1_dx1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        angle = T(6) * T(π) * x[1]
        sin_term = sin(angle)
        cos_term = cos(angle)
        exp_term = exp(-T(4) * x[1])
        return T(4) * exp_term * sin_term^6 - T(36) * T(π) * exp_term * sin_term^5 * cos_term
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = df1_dx1(x)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_x = zero(T)
        @inbounds for i in 2:n
            sum_x += x[i]
        end
        sum_x > zero(T) || throw(DomainError(
            sum_x,
            "ZDT6 Jacobian row 2 requires x₂ + ⋯ + xₙ > 0; " *
            "its x₂, ..., xₙ derivatives are undefined when that sum vanishes.",
        ))

        f1x = f1(x)
        gx = g(x)
        ratio = f1x / gx
        grad[1] = -T(2) * ratio * df1_dx1(x)

        mean_x = sum_x / T(n - 1)
        dg_dxi = T(9) * T(0.25) * mean_x^(-T(0.75)) / T(n - 1)

        term = one(T) + ratio^2
        @inbounds for i in 2:n
            grad[i] = dg_dxi * term
        end
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end
