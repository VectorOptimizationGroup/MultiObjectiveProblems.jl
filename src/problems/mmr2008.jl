"""
    MMR1()

Construct the fixed two-variable, two-objective `MMR1` problem.

The variables are bounded by `0.1 <= x[1] <= 1.0` and
`0.0 <= x[2] <= 1.0`. An analytical Jacobian is registered; objective
Hessians are not registered.
"""
function MMR1()
    meta = META["MMR1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = exp(-((x[2] - T(0.6)) / T(0.4))^2)
        t2 = exp(-((x[2] - T(0.2)) / T(0.04))^2)
        g = T(2) - T(0.8) * t1 - t2
        return g / x[1]
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = exp(-((x[2] - T(0.6)) / T(0.4))^2)
        t2 = exp(-((x[2] - T(0.2)) / T(0.04))^2)
        g = T(2) - T(0.8) * t1 - t2
        grad[1] = -g / x[1]^2
        grad[2] = (T(10) * t1 * (x[2] - T(0.6)) + T(1250) * t2 * (x[2] - T(0.2))) / x[1]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = ([0.1, 0.0], [1.0, 1.0]),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    MMR2()

Construct the fixed two-variable, two-objective `MMR2` problem.

The variables are bounded in `[0, 1]^2`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function MMR2()
    meta = META["MMR2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) + T(10) * x[2]
        faux = x[1] / a
        angle = T(8) * T(π) * x[1]
        return (one(T) - faux^2 - faux * sin(angle)) * a
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) + T(10) * x[2]
        faux = x[1] / a
        angle = T(8) * T(π) * x[1]
        sin_angle = sin(angle)
        cos_angle = cos(angle)

        grad[1] = a * (
            -T(2) * faux / a -
            sin_angle / a -
            T(8) * T(π) * faux * cos_angle
        )
        grad[2] = T(10) * (one(T) - faux^2 - faux * sin_angle) +
                  a * (T(20) * faux * x[1] / a^2 + T(10) * x[1] * sin_angle / a^2)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    MMR3()

Construct the fixed two-variable, two-objective `MMR3` problem.

The variables are bounded in `[-1, 1]^2`. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function MMR3()
    meta = META["MMR3"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^3
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[2] - x[1])^3
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(3) * x[1]^2
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t = (x[2] - x[1])^2
        grad[1] = -T(3) * t
        grad[2] = T(3) * t
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    MMR4()

Construct the fixed three-variable, two-objective `MMR4` problem.

The variables are bounded in `[0, 4]^3`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function MMR4()
    meta = META["MMR4"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        den = T(2) * x[1] + x[2] + T(2) * x[3] + one(T)
        return x[1] - T(2) * x[2] - x[3] - T(36) / den
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return -T(3) * x[1] + x[2] - x[3]
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t = (T(2) * x[1] + x[2] + T(2) * x[3] + one(T))^2
        grad[1] = one(T) + T(72) / t
        grad[2] = -T(2) + T(36) / t
        grad[3] = -one(T) + T(72) / t
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = -T(3)
        grad[2] = one(T)
        grad[3] = -one(T)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), fill(4.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end
