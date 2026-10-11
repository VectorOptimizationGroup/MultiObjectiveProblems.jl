"""
    Far1()

Construct the fixed-dimension `Far1` benchmark with two variables and two
objectives. Each variable is bounded by `[-1, 1]`. An analytical Jacobian is
registered; Hessians are not registered.

The constructor follows the corrected `Far1` transcription cataloged by
Huband et al. (2006), rather than a literal transcription of the equation as
printed by Farina (2002).
"""
function Far1()
    meta = META["Far1"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        term1 = -T(2) * exp(T(15) * (-((x[1] - T(0.1))^2 + x[2]^2)))
        term2 = -exp(T(20) * (-((x[1] - T(0.6))^2 + (x[2] - T(0.6))^2)))
        term3 = exp(T(20) * (-((x[1] + T(0.6))^2 + (x[2] - T(0.6))^2)))
        term4 = exp(T(20) * (-((x[1] - T(0.6))^2 + (x[2] + T(0.6))^2)))
        term5 = exp(T(20) * (-((x[1] + T(0.6))^2 + (x[2] + T(0.6))^2)))
        return term1 + term2 + term3 + term4 + term5
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        term1 = T(2) * exp(T(20) * (-(x[1]^2 + x[2]^2)))
        term2 = exp(T(20) * (-((x[1] - T(0.4))^2 + (x[2] - T(0.6))^2)))
        term3 = -exp(T(20) * (-((x[1] + T(0.5))^2 + (x[2] - T(0.7))^2)))
        term4 = -exp(T(20) * (-((x[1] - T(0.5))^2 + (x[2] + T(0.7))^2)))
        term5 = exp(T(20) * (-((x[1] + T(0.4))^2 + (x[2] + T(0.8))^2)))
        return term1 + term2 + term3 + term4 + term5
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        e1 = exp(T(15) * (-((x[1] - T(0.1))^2 + x[2]^2)))
        e2 = exp(T(20) * (-((x[1] - T(0.6))^2 + (x[2] - T(0.6))^2)))
        e3 = exp(T(20) * (-((x[1] + T(0.6))^2 + (x[2] - T(0.6))^2)))
        e4 = exp(T(20) * (-((x[1] - T(0.6))^2 + (x[2] + T(0.6))^2)))
        e5 = exp(T(20) * (-((x[1] + T(0.6))^2 + (x[2] + T(0.6))^2)))

        grad[1] = T(60) * (x[1] - T(0.1)) * e1 +
                  T(40) * (x[1] - T(0.6)) * e2 -
                  T(40) * (x[1] + T(0.6)) * e3 -
                  T(40) * (x[1] - T(0.6)) * e4 -
                  T(40) * (x[1] + T(0.6)) * e5

        grad[2] = T(60) * x[2] * e1 +
                  T(40) * (x[2] - T(0.6)) * e2 -
                  T(40) * (x[2] - T(0.6)) * e3 -
                  T(40) * (x[2] + T(0.6)) * e4 -
                  T(40) * (x[2] + T(0.6)) * e5
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        e1 = exp(T(20) * (-(x[1]^2 + x[2]^2)))
        e2 = exp(T(20) * (-((x[1] - T(0.4))^2 + (x[2] - T(0.6))^2)))
        e3 = exp(T(20) * (-((x[1] + T(0.5))^2 + (x[2] - T(0.7))^2)))
        e4 = exp(T(20) * (-((x[1] - T(0.5))^2 + (x[2] + T(0.7))^2)))
        e5 = exp(T(20) * (-((x[1] + T(0.4))^2 + (x[2] + T(0.8))^2)))

        grad[1] = -T(80) * x[1] * e1 -
                  T(40) * (x[1] - T(0.4)) * e2 +
                  T(40) * (x[1] + T(0.5)) * e3 +
                  T(40) * (x[1] - T(0.5)) * e4 -
                  T(40) * (x[1] + T(0.4)) * e5

        grad[2] = -T(80) * x[2] * e1 -
                  T(40) * (x[2] - T(0.6)) * e2 +
                  T(40) * (x[2] - T(0.7)) * e3 +
                  T(40) * (x[2] + T(0.7)) * e4 -
                  T(40) * (x[2] + T(0.8)) * e5
        return grad
    end

    bounds = (fill(-1.0, n), fill(1.0, n))

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = bounds,
        jacobian = (df1_dx, df2_dx),
    )
end