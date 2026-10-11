"""
    JOS1(; nvar::Int = 50)

Construct the two-objective `JOS1` problem.

Requires `nvar >= 1`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function JOS1(; nvar::Int = 50)
    n = nvar
    n >= 1 || throw(ArgumentError("nvar must be at least 1 for JOS1"))
    meta = META["JOS1"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_squares = zero(T)
        for i in 1:n
            sum_squares += x[i]^2
        end
        return sum_squares / T(n)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_squares = zero(T)
        for i in 1:n
            sum_squares += (x[i] - T(2))^2
        end
        return sum_squares / T(n)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in 1:n
            grad[i] = T(2) * x[i] / T(n)
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in 1:n
            grad[i] = T(2) * (x[i] - T(2)) / T(n)
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

# In Jin et al.'s F-numbering, F2, F3, and F5 reproduce ZDT1, ZDT2,
# and ZDT3. Those formulations are exposed through the ZDT constructors.

"""
    JOS4(; nvar::Int = 50)

Construct the two-objective `JOS4` problem.

Requires `nvar >= 2`. Each variable is bounded in `[0, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.

At `x[1] == 0`, the second Jacobian row is undefined and its evaluation
throws a `DomainError`. Objective values and the first row remain defined.
"""
function JOS4(; nvar::Int = 50)
    n = nvar
    n >= 2 || throw(ArgumentError("nvar must be at least 2 for JOS4"))
    meta = META["JOS4"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        sum_x2n = zero(T)
        for i in 2:n
            sum_x2n += x[i]
        end
        faux = one(T) + T(9) * sum_x2n / T(n - 1)
        t = x[1] / faux
        return faux * (one(T) - t^T(0.25) - t^T(4))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        x[1] > zero(T) || throw(DomainError(
            x[1],
            "JOS4 Jacobian row 2 requires x₁ > 0; " *
            "its x₁ derivative is undefined at x₁ = 0.",
        ))

        sum_x2n = zero(T)
        for i in 2:n
            sum_x2n += x[i]
        end
        faux = one(T) + T(9) * sum_x2n / T(n - 1)
        t = x[1] / faux

        grad[1] = -T(0.25) * t^T(-0.75) - T(4) * t^T(3)
        for i in 2:n
            grad[i] = T(9) / T(n - 1) * (one(T) - T(0.75) * t^T(0.25) + T(3) * t^T(4))
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end
