"""
    DGO0()

Construct the fixed one-variable, two-objective `DGO0` problem.

The variable is bounded in `[-4, 6]`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function DGO0()
    meta = META["DGO0"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(2))^2
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(2))
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-4.0, n), fill(6.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    DGO1()

Construct the fixed one-variable, two-objective `DGO1` problem.

The variable is bounded in `[-10, 13]`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function DGO1()
    meta = META["DGO1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return sin(x[1])
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return sin(x[1] + T(0.7))
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = cos(x[1])
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = cos(x[1] + T(0.7))
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-10.0, n), fill(13.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    DGO2()

Construct the fixed one-variable, two-objective `DGO2` problem.

The variable is bounded in `[-9, 9]`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function DGO2()
    meta = META["DGO2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(9) - sqrt(T(81) - x[1]^2)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = x[1] / sqrt(T(81) - x[1]^2)
        return grad
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2);
        name = meta.name,
        bounds = (fill(-9.0, n), fill(9.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end