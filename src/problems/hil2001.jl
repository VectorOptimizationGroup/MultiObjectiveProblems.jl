"""
    Hil1()

Construct Hillermeier's fixed two-variable, two-objective academic problem.

The problem has no registered variable bounds. An analytical Jacobian is
registered; objective Hessians are not registered. The objectives are
1-periodic in each variable, so `[0, 1]^2` covers one complete period; it is
the square returned by `recommended_bounds` and not a variable bound.
"""
function Hil1()
    meta = META["Hil1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = T(2) * T(π) / T(360) * (T(45) + T(40) * sin(T(2) * T(π) * x[1]) + T(25) * sin(T(2) * T(π) * x[2]))
        b = one(T) + T(0.5) * cos(T(2) * T(π) * x[1])
        return cos(a) * b
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        a = T(2) * T(π) / T(360) * (T(45) + T(40) * sin(T(2) * T(π) * x[1]) + T(25) * sin(T(2) * T(π) * x[2]))
        b = one(T) + T(0.5) * cos(T(2) * T(π) * x[1])
        return sin(a) * b
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = T(2) * T(π) / T(360) * (T(45) + T(40) * sin(T(2) * T(π) * x[1]) + T(25) * sin(T(2) * T(π) * x[2]))
        b = one(T) + T(0.5) * cos(T(2) * T(π) * x[1])

        grad[1] = -T(160) * T(π)^2 / T(360) * cos(T(2) * T(π) * x[1]) * sin(a) * b -
                  T(π) * sin(T(2) * T(π) * x[1]) * cos(a)
        grad[2] = -T(100) * T(π)^2 / T(360) * cos(T(2) * T(π) * x[2]) * sin(a) * b
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        a = T(2) * T(π) / T(360) * (T(45) + T(40) * sin(T(2) * T(π) * x[1]) + T(25) * sin(T(2) * T(π) * x[2]))
        b = one(T) + T(0.5) * cos(T(2) * T(π) * x[1])

        grad[1] = T(160) * T(π)^2 / T(360) * cos(T(2) * T(π) * x[1]) * cos(a) * b -
                  T(π) * sin(T(2) * T(π) * x[1]) * sin(a)
        grad[2] = T(100) * T(π)^2 / T(360) * cos(T(2) * T(π) * x[2]) * cos(a) * b
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end
