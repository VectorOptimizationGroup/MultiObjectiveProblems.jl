"""
    LE1()

Construct the fixed two-variable, two-objective `LE1` problem.

The variables are bounded in `[-5, 10]^2`. An analytical Jacobian is
registered; objective Hessians are not registered. Both objective values are
defined throughout the box, but the first Jacobian row is undefined at
`x == [0, 0]` and the second at `x == [0.5, 0.5]`. Evaluating an undefined row
throws a `DomainError`.
"""
function LE1()
    meta = META["LE1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return (x[1]^2 + x[2]^2)^T(0.125)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return ((x[1] - T(0.5))^2 + (x[2] - T(0.5))^2)^T(0.25)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        iszero(x[1]) && iszero(x[2]) && throw(DomainError(
            (x[1], x[2]),
            "LE1 Jacobian row 1 is undefined at (0, 0).",
        ))

        t = T(0.25) * (x[1]^2 + x[2]^2)^T(-0.875)
        grad[1] = x[1] * t
        grad[2] = x[2] * t
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        half = T(0.5)
        x[1] == half && x[2] == half && throw(DomainError(
            (x[1], x[2]),
            "LE1 Jacobian row 2 is undefined at (0.5, 0.5).",
        ))

        t = T(0.5) * ((x[1] - T(0.5))^2 + (x[2] - T(0.5))^2)^T(-0.75)
        grad[1] = (x[1] - T(0.5)) * t
        grad[2] = (x[2] - T(0.5)) * t
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-5.0, n), fill(10.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end
