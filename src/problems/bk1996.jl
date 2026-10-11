"""
    BK1()

Construct the fixed two-variable, two-objective `BK1` problem.

The variables are bounded in `[-5, 10]^2`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function BK1()
    meta = META["BK1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + x[2]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(5.0))^2 + (x[2] - T(5.0))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2.0) * x[1]
        grad[2] = T(2.0) * x[2]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2.0) * (x[1] - T(5.0))
        grad[2] = T(2.0) * (x[2] - T(5.0))
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-5.0, n), fill(10.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end