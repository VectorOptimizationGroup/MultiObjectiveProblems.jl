"""
    MLF1()

Construct the fixed one-variable, two-objective `MLF1` problem.

The variable bound is `[0, 20]`. The constructor uses the corrected formulation
reported by Huband et al. An analytical Jacobian is registered; objective
Hessians are not registered.
"""
function MLF1()
    meta = META["MLF1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) + x[1] / T(20)) * sin(x[1])
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (one(T) + x[1] / T(20)) * cos(x[1])
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = sin(x[1]) / T(20) + (one(T) + x[1] / T(20)) * cos(x[1])
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = cos(x[1]) / T(20) - (one(T) + x[1] / T(20)) * sin(x[1])
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (zeros(n), fill(20.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    MLF2()

Construct the fixed two-variable, two-objective `MLF2` problem.

The problem has no explicit variable bounds. Its objectives are the negatives
of the maximization objectives in Molyneaux, Favrat, and Leyland, giving an
equivalent minimization problem. `[-100, 100]^2` is the recommended working
box. An analytical Jacobian is registered; objective Hessians are not
registered.
"""
function MLF2()
    meta = META["MLF2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = x[1]^2 + x[2] - T(11)
        t2 = x[1] + x[2]^2 - T(7)
        return -T(5) + (t1^2 + t2^2) / T(200)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = T(4) * x[1]^2 + T(2) * x[2] - T(11)
        t2 = T(2) * x[1] + T(4) * x[2]^2 - T(7)
        return -T(5) + (t1^2 + t2^2) / T(200)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = x[1]^2 + x[2] - T(11)
        t2 = x[1] + x[2]^2 - T(7)
        grad[1] = (T(2) * x[1] * t1 + t2) / T(100)
        grad[2] = (t1 + T(2) * x[2] * t2) / T(100)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        t1 = T(4) * x[1]^2 + T(2) * x[2] - T(11)
        t2 = T(2) * x[1] + T(4) * x[2]^2 - T(7)
        grad[1] = (T(8) * x[1] * t1 + T(2) * t2) / T(100)
        grad[2] = (T(2) * t1 + T(8) * x[2] * t2) / T(100)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end
