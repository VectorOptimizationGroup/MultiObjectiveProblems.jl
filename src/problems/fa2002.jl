"""
    FA1()

Construct the fixed three-variable, three-objective `FA1` problem.

The variables are bounded in `[0, 1]^3`. An analytical Jacobian is registered;
objective Hessians are not registered. The objective values are defined at
`x[1] == 0`, but the second and third Jacobian rows are not; evaluating either
of those rows there throws a `DomainError`. The first Jacobian row remains
available at that boundary.
"""
function FA1()
    meta = META["FA1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) - exp(-T(4) * x[1])) / (one(T) - exp(-T(4)))
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        num = (one(T) - exp(-T(4) * x[1])) / (one(T) - exp(-T(4)))
        den = x[2] + one(T)
        return den * (one(T) - (num / den)^T(0.5))
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        num = (one(T) - exp(-T(4) * x[1])) / (one(T) - exp(-T(4)))
        den = x[3] + one(T)
        return den * (one(T) - (num / den)^T(0.1))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(4) * exp(-T(4) * x[1]) / (one(T) - exp(-T(4)))
        grad[2] = zero(T)
        grad[3] = zero(T)
        return grad
    end

    check_jacobian_domain = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        x[1] > zero(T) || throw(DomainError(
            x[1],
            "FA1 Jacobian rows 2 and 3 require x₁ > 0; " *
            "their x₁ derivatives are undefined at x₁ = 0.",
        ))
        return nothing
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        check_jacobian_domain(x)
        a = exp(-T(4) * x[1])
        scale = one(T) - exp(-T(4))
        s = ((one(T) - a) / scale) / (x[2] + one(T))
        grad[1] = -T(2) * a / scale * s^(-T(0.5))
        grad[2] = one(T) - T(0.5) * s^T(0.5)
        grad[3] = zero(T)
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        check_jacobian_domain(x)
        a = exp(-T(4) * x[1])
        scale = one(T) - exp(-T(4))
        s = ((one(T) - a) / scale) / (x[3] + one(T))
        grad[1] = -T(0.4) * a / scale * s^(-T(0.9))
        grad[2] = zero(T)
        grad[3] = one(T) - T(0.9) * s^T(0.1)
        return grad
    end

    bounds = (fill(0.0, n), fill(1.0, n))

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = bounds,
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
