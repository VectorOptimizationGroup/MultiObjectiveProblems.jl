"""
    Lov1()

Create the first Lovison problem with two variables and two objectives.

No variable bounds are registered. An analytical Jacobian is registered;
Hessians are not registered. The catalog metadata classifies both objectives
as strictly convex.
"""
function Lov1()
    meta = META["Lov1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(1.05) * x[1]^2 + T(0.98) * x[2]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(0.99) * (x[1] - T(3))^2 + T(1.03) * (x[2] - T(2.5))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2.1) * x[1]
        grad[2] = T(1.96) * x[2]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(1.98) * (x[1] - T(3))
        grad[2] = T(2.06) * (x[2] - T(2.5))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Lov2()

Create the second Lovison problem with two variables and two objectives.

No variable bounds are registered. The second objective is singular at
`x[1] == -1`. An analytical Jacobian is registered; Hessians are not
registered. The catalog metadata classifies both objectives as not strictly
convex.
"""
function Lov2()
    meta = META["Lov2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[2]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return -((x[2] - x[1]^3) / (x[1] + one(T)))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = zero(T)
        grad[2] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -((-T(3) * x[1]^2 * (x[1] + one(T)) - (x[2] - x[1]^3)) / (x[1] + one(T))^2)
        grad[2] = -one(T) / (x[1] + one(T))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Lov3()

Create the third Lovison problem with two variables and two objectives.

No variable bounds are registered. An analytical Jacobian is registered;
Hessians are not registered. The catalog metadata classifies the first
objective as strictly convex and the second as not strictly convex.
"""
function Lov3()
    meta = META["Lov3"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + x[2]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(6))^2 - (x[2] + T(0.3))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = T(2) * x[2]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(6))
        grad[2] = -T(2) * (x[2] + T(0.3))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Lov4()

Create the fourth Lovison problem with two variables and two objectives.

No variable bounds are registered. An analytical Jacobian is registered;
Hessians are not registered. The catalog metadata classifies the first
objective as not strictly convex and the second as strictly convex.
"""
function Lov4()
    meta = META["Lov4"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        exp1 = exp(-(x[1] + T(2))^2 - x[2]^2)
        exp2 = exp(-(x[1] - T(2))^2 - x[2]^2)
        return x[1]^2 + x[2]^2 + T(4) * (exp1 + exp2)
    end
    
    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(6))^2 + (x[2] + T(0.5))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp1 = exp(-(x[1] + T(2))^2 - x[2]^2)
        exp2 = exp(-(x[1] - T(2))^2 - x[2]^2)
        
        grad[1] = T(2) * x[1] - T(8) * ((x[1] + T(2)) * exp1 + (x[1] - T(2)) * exp2)
        grad[2] = T(2) * x[2] - T(8) * (x[2] * exp1 + x[2] * exp2)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(6))
        grad[2] = T(2) * (x[2] + T(0.5))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Lov5()

Create the fifth Lovison problem with three variables and two objectives.

No variable bounds are registered. An analytical Jacobian is registered;
Hessians are not registered. The catalog metadata classifies both objectives
as not strictly convex.
"""
function Lov5()
    meta = META["Lov5"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    g = function (x::T, y::T, z::T, px::T, py::T, pz::T, σ::T) where {T <: AbstractFloat}
        m11 = -one(T)
        m12 = T(-0.03)
        m13 = T(0.011)
        m22 = -one(T)
        m23 = T(0.07)
        m33 = T(-1.01)

        vx = x - px
        vy = y - py
        vz = z - pz
        q = m11 * vx^2 + T(2) * m12 * vx * vy + T(2) * m13 * vx * vz +
            m22 * vy^2 + T(2) * m23 * vy * vz + m33 * vz^2

        return sqrt(T(2) * T(π) / σ) * exp(q / σ^2)
    end

    f = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return g(x[1], x[2], x[3], zero(T), T(0.15), zero(T), T(0.35)) +
               g(x[1], x[2], T(0.5) * x[3], zero(T), T(-1.1), zero(T), T(3))
    end

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return -sqrt(T(2)) / T(2) * (x[1] + f(x))
    end
    
    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return -sqrt(T(2)) / T(2) * (-x[1] + f(x))
    end

    df_dx = function (df::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        m11 = -one(T)
        m12 = T(-0.03)
        m13 = T(0.011)
        m22 = -one(T)
        m23 = T(0.07)
        m33 = T(-1.01)
        σ0 = T(0.35)
        σ1 = T(3)

        v0x = x[1]
        v0y = x[2] - T(0.15)
        v0z = x[3]
        g0 = g(x[1], x[2], x[3], zero(T), T(0.15), zero(T), σ0)
        dg0_dx = g0 * T(2) * (m11 * v0x + m12 * v0y + m13 * v0z) / σ0^2
        dg0_dy = g0 * T(2) * (m12 * v0x + m22 * v0y + m23 * v0z) / σ0^2
        dg0_dz = g0 * T(2) * (m13 * v0x + m23 * v0y + m33 * v0z) / σ0^2

        v1x = x[1]
        v1y = x[2] + T(1.1)
        v1z = T(0.5) * x[3]
        g1 = g(x[1], x[2], T(0.5) * x[3], zero(T), T(-1.1), zero(T), σ1)
        dg1_dx = g1 * T(2) * (m11 * v1x + m12 * v1y + m13 * v1z) / σ1^2
        dg1_dy = g1 * T(2) * (m12 * v1x + m22 * v1y + m23 * v1z) / σ1^2
        dg1_dz = g1 * (m13 * v1x + m23 * v1y + m33 * v1z) / σ1^2

        df[1] = dg0_dx + dg1_dx
        df[2] = dg0_dy + dg1_dy
        df[3] = dg0_dz + dg1_dz
        return df
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        df_dx(grad, x)
        scale = sqrt(T(2)) / T(2)
        grad[1] = -scale * (one(T) + grad[1])
        grad[2] *= -scale
        grad[3] *= -scale
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        df_dx(grad, x)
        scale = sqrt(T(2)) / T(2)
        grad[1] = -scale * (-one(T) + grad[1])
        grad[2] *= -scale
        grad[3] *= -scale
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Lov6()

Create the sixth Lovison problem with six variables and two objectives.

The first variable is bounded by `[0.1, 0.425]`; the remaining variables are
bounded by `[-0.16, 0.16]`. An analytical Jacobian is registered; Hessians are
not registered. The catalog metadata classifies both objectives as not
strictly convex.
"""
function Lov6()
    meta = META["Lov6"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end
    
    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return one(T) - sqrt(x[1]) - x[1] * sin(T(10) * π * x[1]) +
               x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[6]^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        fill!(grad, zero(T))
        grad[1] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -T(0.5) / sqrt(x[1]) - sin(T(10) * π * x[1]) - T(10) * π * x[1] * cos(T(10) * π * x[1])
        for i in 2:6
            grad[i] = T(2) * x[i]
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = ([0.1, -0.16, -0.16, -0.16, -0.16, -0.16], [0.425, 0.16, 0.16, 0.16, 0.16, 0.16]),
        jacobian = (df1_dx, df2_dx),
    )
end
