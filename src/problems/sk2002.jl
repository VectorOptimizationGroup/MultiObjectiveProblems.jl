"""
    SK1()

Construct the fixed one-variable, two-objective `SK1` problem.

Socha and Kisiel-Dorohinicki formulate the problem as a maximization; this
constructor minimizes the negated objectives, using the corrected second
objective cataloged by Huband et al. The problem has no explicit variable
bounds; `[-100, 100]` is the box recommended by the package developers, not a
bound of the problem. An analytical Jacobian is registered; objective Hessians are not
registered.
"""
function SK1()
    meta = META["SK1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^4 + T(3) * x[1]^3 - T(10) * x[1]^2 - T(10) * x[1] - T(10)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(0.5) * x[1]^4 - T(2) * x[1]^3 - T(10) * x[1]^2 + T(10) * x[1] - T(5)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(4) * x[1]^3 + T(9) * x[1]^2 - T(20) * x[1] - T(10)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]^3 - T(6) * x[1]^2 - T(20) * x[1] + T(10)
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    SK2()

Construct the fixed four-variable, two-objective `SK2` problem.

Socha and Kisiel-Dorohinicki formulate the problem as a maximization; this
constructor minimizes the negated objectives. The problem has no explicit
variable bounds; `[-10, 10]^4` is the box recommended by the package
developers, not a bound of the problem. An analytical Jacobian is registered; objective Hessians
are not registered.
"""
function SK2()
    meta = META["SK2"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1] - T(2))^2 + (x[2] + T(3))^2 + (x[3] - T(5))^2 + (x[4] - T(4))^2 - T(5)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        numer = sin(x[1]) + sin(x[2]) + sin(x[3]) + sin(x[4])
        denom = one(T) + (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) / T(100)
        return -numer / denom
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * (x[1] - T(2))
        grad[2] = T(2) * (x[2] + T(3))
        grad[3] = T(2) * (x[3] - T(5))
        grad[4] = T(2) * (x[4] - T(4))
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        denom = one(T) + (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) / T(100)
        numer = sin(x[1]) + sin(x[2]) + sin(x[3]) + sin(x[4])
        common = numer / T(50)
        denom2 = denom^2
        grad[1] = (-cos(x[1]) * denom + common * x[1]) / denom2
        grad[2] = (-cos(x[2]) * denom + common * x[2]) / denom2
        grad[3] = (-cos(x[3]) * denom + common * x[3]) / denom2
        grad[4] = (-cos(x[4]) * denom + common * x[4]) / denom2
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        jacobian = (df1_dx, df2_dx),
    )
end