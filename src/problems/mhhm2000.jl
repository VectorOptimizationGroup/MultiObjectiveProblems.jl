"""
    MHHM1()

Return the fixed-dimension MHHM1 problem with one variable and three objectives.

The variable is bounded by `[0, 1]`. An analytical Jacobian is registered, but
Hessians are not. The catalog metadata classifies all three objectives as
strictly convex. The default dimensions are `nvar = 1` and `nobj = 3`.
"""
function MHHM1()
    meta = META["MHHM1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.8))^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.85))^2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.9))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.8))
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.85))
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.9))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end

"""
    MHHM2()

Return the fixed-dimension MHHM2 problem with two variables and three objectives.

Each variable is bounded by `[0, 1]`. An analytical Jacobian is registered, but
Hessians are not. The catalog metadata classifies all three objectives as
strictly convex. The default dimensions are `nvar = 2` and `nobj = 3`.
"""
function MHHM2()
    meta = META["MHHM2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.8))^2 + (x[2] - T(0.6))^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.85))^2 + (x[2] - T(0.7))^2
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(0.9))^2 + (x[2] - T(0.6))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.8))
        grad[2] = T(2) * (x[2] - T(0.6))
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.85))
        grad[2] = T(2) * (x[2] - T(0.7))
        return grad
    end

    df3_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(0.9))
        grad[2] = T(2) * (x[2] - T(0.6))
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2, f3);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end
