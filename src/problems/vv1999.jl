"""
    MOP2(; nvar::Int = 3)

Construct the two-objective `MOP2` problem.

Requires `nvar >= 1`. Each variable is bounded in `[-4, 4]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function MOP2(; nvar::Int = 3)
    n = nvar
    n >= 1 || throw(ArgumentError("nvar must be at least 1 for MOP2"))
    meta = META["MOP2"]
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) / sqrt(T(n))
        s = zero(T)
        @inbounds for i in 1:n
            s += (x[i] - a)^2
        end
        return one(T) - exp(-s)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) / sqrt(T(n))
        s = zero(T)
        @inbounds for i in 1:n
            s += (x[i] + a)^2
        end
        return one(T) - exp(-s)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) / sqrt(T(n))
        s = zero(T)
        @inbounds for i in 1:n
            s += (x[i] - a)^2
        end
        fac = exp(-s)
        @inbounds for i in 1:n
            grad[i] = T(2) * (x[i] - a) * fac
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) / sqrt(T(n))
        s = zero(T)
        @inbounds for i in 1:n
            s += (x[i] + a)^2
        end
        fac = exp(-s)
        @inbounds for i in 1:n
            grad[i] = T(2) * (x[i] + a) * fac
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-4.0, n), fill(4.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

# ------------------------- MOP3 -------------------------
"""
    MOP3()

Construct the fixed two-variable, two-objective `MOP3` problem.

Poloni et al. and Van Veldhuizen formulate the problem as a maximization; this
constructor minimizes the negated objectives. The Pareto-optimal decision set
is preserved, while reported objective values are sign-reversed. The variables
are bounded in `[-π, π]^2`. An analytical Jacobian is registered; objective
Hessians are not registered.
"""
function MOP3()
    meta = META["MOP3"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        A1 = T(0.5) * sin(one(T)) - T(2) * cos(one(T)) + sin(T(2)) - T(1.5) * cos(T(2))
        A2 = T(1.5) * sin(one(T)) - cos(one(T)) + T(2) * sin(T(2)) - T(0.5) * cos(T(2))
        B1 = T(0.5) * sin(x[1]) - T(2) * cos(x[1]) + sin(x[2]) - T(1.5) * cos(x[2])
        B2 = T(1.5) * sin(x[1]) - cos(x[1]) + T(2) * sin(x[2]) - T(0.5) * cos(x[2])
        return one(T) + (A1 - B1)^2 + (A2 - B2)^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] + T(3))^2 + (x[2] + one(T))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        A1 = T(0.5) * sin(one(T)) - T(2) * cos(one(T)) + sin(T(2)) - T(1.5) * cos(T(2))
        A2 = T(1.5) * sin(one(T)) - cos(one(T)) + T(2) * sin(T(2)) - T(0.5) * cos(T(2))
        B1 = T(0.5) * sin(x[1]) - T(2) * cos(x[1]) + sin(x[2]) - T(1.5) * cos(x[2])
        B2 = T(1.5) * sin(x[1]) - cos(x[1]) + T(2) * sin(x[2]) - T(0.5) * cos(x[2])
        grad[1] = T(2) * (A1 - B1) * (-T(0.5) * cos(x[1]) - T(2) * sin(x[1])) +
                  T(2) * (A2 - B2) * (-T(1.5) * cos(x[1]) - sin(x[1]))
        grad[2] = T(2) * (A1 - B1) * (-cos(x[2]) - T(1.5) * sin(x[2])) +
                  T(2) * (A2 - B2) * (-T(2) * cos(x[2]) - T(0.5) * sin(x[2]))
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] + T(3))
        grad[2] = T(2) * (x[2] + one(T))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-π, n), fill(π, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

# ------------------------- MOP5 -------------------------
"""
    MOP5()

Construct the fixed two-variable, three-objective `MOP5` problem.

The variables are bounded in `[-30, 30]^2`. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function MOP5()
    meta = META["MOP5"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        r2 = x[1]^2 + x[2]^2
        return T(0.5) * r2 + sin(r2)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (T(3) * x[1] - T(2) * x[2] + T(4))^2 / T(8) +
               (x[1] - x[2] + one(T))^2 / T(27) + T(15)
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        r2 = x[1]^2 + x[2]^2
        return one(T) / (r2 + one(T)) - T(1.1) * exp(-r2)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        c = cos(x[1]^2 + x[2]^2)
        grad[1] = x[1] + T(2) * x[1] * c
        grad[2] = x[2] + T(2) * x[2] * c
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(3) * (T(3) * x[1] - T(2) * x[2] + T(4)) / T(4) +
                  T(2) * (x[1] - x[2] + one(T)) / T(27)
        grad[2] = -T(2) * (T(3) * x[1] - T(2) * x[2] + T(4)) / T(4) -
                  T(2) * (x[1] - x[2] + one(T)) / T(27)
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        r2 = x[1]^2 + x[2]^2
        denom = (r2 + one(T))^2
        e = exp(-r2)
        grad[1] = -T(2) * x[1] / denom + T(2.2) * x[1] * e
        grad[2] = -T(2) * x[2] / denom + T(2.2) * x[2] * e
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-30.0, n), fill(30.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end

# ------------------------- MOP6 -------------------------
"""
    MOP6(; q::Int = 4)

Construct the fixed two-variable, two-objective `MOP6` problem.

`q` is the frequency of the trigonometric term and must be at least 1; the
source describes the problem as scalable in the number of Pareto curves, which
`q` controls, and uses `q = 4`. The variables are bounded in `[0, 1]^2`. An
analytical Jacobian is registered; objective Hessians are not registered.
"""
function MOP6(; q::Int = 4)
    q >= 1 || throw(ArgumentError("q must be at least 1 for MOP6"))
    meta = META["MOP6"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) + T(10) * x[2]
        t = x[1] / a
        return a * (one(T) - t^2 - t * sin(T(2) * T(π) * T(q) * x[1]))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = one(T) + T(10) * x[2]
        t = x[1] / a
        ω = T(2) * T(π) * T(q)
        angle = ω * x[1]
        b = sin(angle)
        grad[1] = -T(2) * t - b - ω * x[1] * cos(angle)
        grad[2] = T(10) * (one(T) - t^2 - t * b) + T(10) * x[1] / a * (T(2) * t + b)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx),
    )
end

# ------------------------- MOP7 -------------------------
"""
    MOP7()

Construct the fixed two-variable, three-objective `MOP7` problem.

The variables are bounded in `[-400, 400]^2`. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function MOP7()
    meta = META["MOP7"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(2))^2 / T(2) + (x[2] + one(T))^2 / T(13) + T(3)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] + x[2] - T(3))^2 / T(36) + (-x[1] + x[2] + T(2))^2 / T(8) - T(17)
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] + T(2) * x[2] - one(T))^2 / T(175) +
               (-x[1] + T(2) * x[2])^2 / T(17) - T(13)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = x[1] - T(2)
        grad[2] = T(2) * (x[2] + one(T)) / T(13)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = (x[1] + x[2] - T(3)) / T(18) - (-x[1] + x[2] + T(2)) / T(4)
        grad[2] = (x[1] + x[2] - T(3)) / T(18) + (-x[1] + x[2] + T(2)) / T(4)
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] + T(2) * x[2] - one(T)) / T(175) -
                  T(2) * (-x[1] + T(2) * x[2]) / T(17)
        grad[2] = T(4) * (x[1] + T(2) * x[2] - one(T)) / T(175) +
                  T(4) * (-x[1] + T(2) * x[2]) / T(17)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-400.0, n), fill(400.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
