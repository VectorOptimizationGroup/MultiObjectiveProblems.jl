"""
    SLCDT1(; lambda::Real = 0.85)

Construct the fixed two-variable, two-objective `SLCDT1` problem, with
perturbation coefficient `lambda`.

The variables are bounded in `[-0.5, 0.5]^2` when `lambda == 0`, and in
`[-1.5, 1.5]^2` for every other value of `lambda` (including the default). An
analytical Jacobian is registered; objective Hessians are not registered.
"""
function SLCDT1(; lambda::Real = 0.85)
    meta = META["SLCDT1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = x[1] + x[2]
        d = x[1] - x[2]
        return T(0.5) * (sqrt(one(T) + s^2) + sqrt(one(T) + d^2) + x[1] - x[2]) +
               T(lambda) * exp(-d^2)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = x[1] + x[2]
        d = x[1] - x[2]
        return T(0.5) * (sqrt(one(T) + s^2) + sqrt(one(T) + d^2) - x[1] + x[2]) +
               T(lambda) * exp(-d^2)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        s = x[1] + x[2]
        d = x[1] - x[2]
        t1 = s / sqrt(one(T) + s^2)
        t2 = d / sqrt(one(T) + d^2)
        pert = -T(2) * T(lambda) * d * exp(-d^2)
        grad[1] = T(0.5) * (t1 + t2 + one(T)) + pert
        grad[2] = T(0.5) * (t1 - t2 - one(T)) - pert
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        s = x[1] + x[2]
        d = x[1] - x[2]
        t1 = s / sqrt(one(T) + s^2)
        t2 = d / sqrt(one(T) + d^2)
        pert = -T(2) * T(lambda) * d * exp(-d^2)
        grad[1] = T(0.5) * (t1 + t2 - one(T)) + pert
        grad[2] = T(0.5) * (t1 - t2 + one(T)) - pert
        return grad
    end

    lo, hi = lambda == 0 ? (-0.5, 0.5) : (-1.5, 1.5)

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(lo, n), fill(hi, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    SLCDT2()

Construct the fixed ten-variable, three-objective `SLCDT2` problem.

The variables are bounded in `[-1, 1]^10`. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function SLCDT2()
    meta = META["SLCDT2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = (x[1] - one(T))^4
        @inbounds for i in 2:n
            s += (x[i] - one(T))^2
        end
        return s
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = (x[2] + one(T))^4
        @inbounds for i in 1:n
            if i != 2
                s += (x[i] + one(T))^2
            end
        end
        return s
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = (x[3] - one(T))^4
        @inbounds for i in 1:n
            if i != 3
                alt = isodd(i) ? one(T) : -one(T)
                s += (x[i] - alt)^2
            end
        end
        return s
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(4) * (x[1] - one(T))^3
        @inbounds for i in 2:n
            grad[i] = T(2) * (x[i] - one(T))
        end
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        @inbounds for i in 1:n
            if i == 2
                grad[i] = T(4) * (x[i] + one(T))^3
            else
                grad[i] = T(2) * (x[i] + one(T))
            end
        end
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        @inbounds for i in 1:n
            if i == 3
                grad[i] = T(4) * (x[i] - one(T))^3
            else
                alt = isodd(i) ? one(T) : -one(T)
                grad[i] = T(2) * (x[i] - alt)
            end
        end
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
