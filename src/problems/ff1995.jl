"""
    FF1()

Construct the fixed two-variable, two-objective `FF1` problem.

The problem has no explicit variable bounds. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function FF1()
    meta = META["FF1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return one(T) - exp(-((x[1] - one(T))^2 + (x[2] + one(T))^2))
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return one(T) - exp(-((x[1] + one(T))^2 + (x[2] - one(T))^2))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp_term = exp(-((x[1] - one(T))^2 + (x[2] + one(T))^2))
        grad[1] = T(2) * (x[1] - one(T)) * exp_term
        grad[2] = T(2) * (x[2] + one(T)) * exp_term
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        exp_term = exp(-((x[1] + one(T))^2 + (x[2] - one(T))^2))
        grad[1] = T(2) * (x[1] + one(T)) * exp_term
        grad[2] = T(2) * (x[2] - one(T)) * exp_term
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end
