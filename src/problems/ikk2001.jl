"""
    IKK1()

Construct the fixed two-variable, three-objective `IKK1` problem.

The variables are bounded in `[-50, 50]^2`. An analytical Jacobian is
registered; objective Hessians are not registered.
"""
function IKK1()
    meta = META["IKK1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(20))^2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[2]^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(20))
        grad[2] = zero(T)
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = zero(T)
        grad[2] = T(2) * x[2]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (fill(-50.0, n), fill(50.0, n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
