"""
    VU1()

Construct the fixed two-variable, two-objective `VU1` problem.

The variables are bounded in `[-3, 3]^2`. An analytical Jacobian is registered;
objective Hessians are not registered.

The constructor uses the `VU1` formulation cataloged in Table XVI of Huband et
al. (2006), since Valenzuela-Rendón and Uresti-Charre (1997) do not state the
objectives in a directly verifiable form.
"""
function VU1()
    meta = META["VU1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return one(T) / (x[1]^2 + x[2]^2 + one(T))
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + T(3) * x[2]^2 + one(T)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        den = x[1]^2 + x[2]^2 + one(T)
        coeff = -T(2) / den^2
        grad[1] = coeff * x[1]
        grad[2] = coeff * x[2]
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = T(6) * x[2]
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-3.0, n), fill(3.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    VU2()

Construct the fixed two-variable, two-objective `VU2` problem.

The variables are bounded in `[-3, 3]^2`. An analytical Jacobian is registered;
objective Hessians are not registered.

The constructor uses the `VU2` formulation cataloged in Table XVI of Huband et
al. (2006), since Valenzuela-Rendón and Uresti-Charre (1997) do not state the
objectives in a directly verifiable form.
"""
function VU2()
    meta = META["VU2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1] + x[2] + one(T)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + T(2) * x[2] - one(T)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = one(T)
        grad[2] = one(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = T(2)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-3.0, n), fill(3.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end