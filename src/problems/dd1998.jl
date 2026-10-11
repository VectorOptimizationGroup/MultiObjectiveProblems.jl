"""
    DD1()

Construct the fixed five-variable, two-objective `DD1` problem.

The problem has two equality constraints and one inequality constraint. It has
no explicit variable bounds. Analytical Jacobians and Hessians are registered
for both the objectives and the constraints.
"""
function DD1()
    meta = META["DD1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        acc = zero(T)
        for xi in x
            acc += xi^2
        end
        return acc
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        d = x[4] - x[5]
        return T(3) * x[1] + T(2) * x[2] - x[3] / T(3) + T(0.01) * d^3
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in eachindex(x)
            grad[i] = T(2) * x[i]
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        d2 = (x[4] - x[5])^2
        grad[1] = T(3)
        grad[2] = T(2)
        grad[3] = -one(T) / T(3)
        grad[4] = T(0.03) * d2
        grad[5] = -T(0.03) * d2
        return grad
    end

    d2f1_dx2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        for i in axes(H, 1)
            H[i, i] = T(2)
        end
        return H
    end

    d2f2_dx2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        curvature = T(0.06) * (x[4] - x[5])
        H[4, 4] = curvature
        H[4, 5] = -curvature
        H[5, 4] = -curvature
        H[5, 5] = curvature
        return H
    end

    c1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1] + T(2) * x[2] - x[3] - T(0.5) * x[4] + x[5] - T(2)
    end

    c2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(4) * x[1] - T(2) * x[2] + T(0.8) * x[3] +
               T(0.6) * x[4] + T(0.5) * x[5]^2
    end

    c3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return f1(x) - T(10)
    end

    dc1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = T(2)
        grad[3] = -one(T)
        grad[4] = -T(0.5)
        grad[5] = one(T)
        return grad
    end

    dc2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(4)
        grad[2] = -T(2)
        grad[3] = T(0.8)
        grad[4] = T(0.6)
        grad[5] = x[5]
        return grad
    end

    dc3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        for i in eachindex(x)
            grad[i] = T(2) * x[i]
        end
        return grad
    end

    d2c1_dx2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        return H
    end

    d2c2_dx2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        H[5, 5] = one(T)
        return H
    end

    d2c3_dx2 = function (H::AbstractMatrix{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(H, zero(T))
        for i in axes(H, 1)
            H[i, i] = T(2)
        end
        return H
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
        hessian = (d2f1_dx2, d2f2_dx2),
        c = (c1, c2, c3),
        lcon = (0.0, 0.0, -Inf),
        ucon = (0.0, 0.0, 0.0),
        constraint_jacobian = (dc1_dx, dc2_dx, dc3_dx),
        constraint_hessian = (d2c1_dx2, d2c2_dx2, d2c3_dx2),
    )
end