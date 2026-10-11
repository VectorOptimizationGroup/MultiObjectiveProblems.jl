"""
    LTDZ1()

Construct the fixed-dimension `LTDZ1` benchmark with three variables and three
objectives. Each variable is bounded by `[0, 1]`. An analytical Jacobian is
registered; objective Hessians are not registered.

The constructor uses the third objective given for `LTDZ1` in Table XVI of
Huband et al. (2006), rather than a literal transcription of Equation (8) in
Laumanns et al. (2002). It negates all three maximization objectives to follow
the package's minimization convention.
"""
function LTDZ1()
    meta = META["LTDZ1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        b = x[2] * halfpi
        return T(-3) + (one(T) + x[3]) * cos(a) * cos(b)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        b = x[2] * halfpi
        return T(-3) + (one(T) + x[3]) * cos(a) * sin(b)
    end

    f3 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        return T(-3) + (one(T) + x[3]) * sin(a)
    end

    df1_dx = function (g::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        b = x[2] * halfpi
        g[1] = -halfpi * (one(T) + x[3]) * sin(a) * cos(b)
        g[2] = -halfpi * (one(T) + x[3]) * cos(a) * sin(b)
        g[3] = cos(a) * cos(b)
        return g
    end

    df2_dx = function (g::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        b = x[2] * halfpi
        g[1] = -halfpi * (one(T) + x[3]) * sin(a) * sin(b)
        g[2] = halfpi * (one(T) + x[3]) * cos(a) * cos(b)
        g[3] = cos(a) * sin(b)
        return g
    end

    df3_dx = function (g::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        halfpi = T(π) / T(2)
        a = x[1] * halfpi
        g[1] = halfpi * (one(T) + x[3]) * cos(a)
        g[2] = zero(T)
        g[3] = sin(a)
        return g
    end

    return MultiObjectiveProblem(
        n,
        m,
        (f1, f2, f3);
        name = meta.name,
        bounds = (zeros(n), ones(n)),
        jacobian = (df1_dx, df2_dx, df3_dx),
    )
end

"""
    LTDZ()

Alias for `LTDZ1()`. Fliege, Drummond, and Svaiter (2009) use the name `LTDZ`
for this benchmark, while Huband et al. (2006) catalog it as `LTDZ1`.
"""
LTDZ() = LTDZ1()
