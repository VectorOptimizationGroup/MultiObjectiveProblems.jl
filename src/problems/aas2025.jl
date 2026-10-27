"""
    AAS1()

Construct the fixed two-variable, two-objective `AAS1` problem.

The variables are bounded in `[-2, 2]^2`. An analytical Jacobian is registered;
objective Hessians are not registered. Since `p > 1`, the gradient of `f₂` is
defined where a component of `Dx` vanishes, and that component contributes
zero there.
"""
function AAS1()
    meta = META["AAS1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        r1 = T(2.0) * x[1] + T(0.5) * x[2] - one(T)
        r2 = T(0.5) * x[1] + T(1.5) * x[2] + T(0.5)
        return T(0.5) * (r1 * r1 + r2 * r2)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.003)
        λ = T(0.9)
        y1 = x[1] + T(0.8) * x[2]
        y2 = T(0.3) * x[1] + T(1.2) * x[2]
        return (λ / p) * (abs(y1)^p + abs(y2)^p)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        r1 = T(2.0) * x[1] + T(0.5) * x[2] - one(T)
        r2 = T(0.5) * x[1] + T(1.5) * x[2] + T(0.5)
        grad[1] = T(2.0) * r1 + T(0.5) * r2
        grad[2] = T(0.5) * r1 + T(1.5) * r2
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.003)
        λ = T(0.9)
        y1 = x[1] + T(0.8) * x[2]
        y2 = T(0.3) * x[1] + T(1.2) * x[2]
        s1 = sign(y1) * abs(y1)^(p - one(T))
        s2 = sign(y2) * abs(y2)^(p - one(T))
        grad[1] = λ * (s1 + T(0.3) * s2)
        grad[2] = λ * (T(0.8) * s1 + T(1.2) * s2)
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds=(fill(-2.0, n), fill(2.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    AAS2()

Construct the fixed two-variable, two-objective `AAS2` problem.

The variables are bounded in `[-5, 5]^2`. An analytical Jacobian is registered;
objective Hessians are not registered. Since `p₁, p₂ > 1`, the gradient of each
objective is defined where a component of `Dᵢ(x - cᵢ)` vanishes, and that
component contributes zero there.
"""
function AAS2()
    meta = META["AAS2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.003)
        λ = T(1.2)
        u1 = x[1] - T(1.5)
        u2 = x[2] + one(T)
        y1 = T(1.2) * u1 - T(0.3) * u2
        y2 = T(0.4) * u1 + T(1.5) * u2
        return (λ / p) * (abs(y1)^p + abs(y2)^p)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.002)
        λ = T(0.8)
        u1 = x[1] + T(1.2)
        u2 = x[2] - T(0.8)
        y1 = T(1.8) * u1 + T(0.5) * u2
        y2 = -T(0.2) * u1 + T(1.1) * u2
        return (λ / p) * (abs(y1)^p + abs(y2)^p)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.003)
        λ = T(1.2)
        u1 = x[1] - T(1.5)
        u2 = x[2] + one(T)
        y1 = T(1.2) * u1 - T(0.3) * u2
        y2 = T(0.4) * u1 + T(1.5) * u2
        s1 = sign(y1) * abs(y1)^(p - one(T))
        s2 = sign(y2) * abs(y2)^(p - one(T))
        grad[1] = λ * (T(1.2) * s1 + T(0.4) * s2)
        grad[2] = λ * (-T(0.3) * s1 + T(1.5) * s2)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        p = T(1.002)
        λ = T(0.8)
        u1 = x[1] + T(1.2)
        u2 = x[2] - T(0.8)
        y1 = T(1.8) * u1 + T(0.5) * u2
        y2 = -T(0.2) * u1 + T(1.1) * u2
        s1 = sign(y1) * abs(y1)^(p - one(T))
        s2 = sign(y2) * abs(y2)^(p - one(T))
        grad[1] = λ * (T(1.8) * s1 - T(0.2) * s2)
        grad[2] = λ * (T(0.5) * s1 + T(1.1) * s2)
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds=(fill(-5.0, n), fill(5.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end