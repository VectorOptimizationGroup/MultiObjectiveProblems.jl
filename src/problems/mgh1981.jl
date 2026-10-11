# The names retain the numbering of the nonlinear least-squares residuals from
# Moré, Garbow, and Hillstrom (1981). The objectives follow the multiobjective
# adaptations of Mita, Fukuda, and Yamashita (2019), with documented extensions
# to their experimental dimensions.

"""
    MGH9()

Return the fixed Gaussian instance with 3 variables and 15 objectives. Variables
are bounded by `[-2, 2]`. An analytical Jacobian is registered; Hessians are not.
"""
function MGH9()
    meta = META["MGH9"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    tdata = ntuple(i -> (8.0 - i) / 2.0, m)
    ydata = (
        9.0e-4,
        4.4e-3,
        1.75e-2,
        5.4e-2,
        1.295e-1,
        2.42e-1,
        3.521e-1,
        3.989e-1,
        3.521e-1,
        2.42e-1,
        1.295e-1,
        5.4e-2,
        1.75e-2,
        4.4e-3,
        9.0e-4,
    )

    objectives = ntuple(i -> function (x::AbstractVector{T}) where {T <: AbstractFloat}
        z = T(tdata[i]) - x[3]
        return x[1] * exp(-x[2] * z^2 / T(2)) - T(ydata[i])
    end, m)

    gradients = ntuple(i -> function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        z = T(tdata[i]) - x[3]
        e = exp(-x[2] * z^2 / T(2))
        grad[1] = e
        grad[2] = -x[1] * e * z^2 / T(2)
        grad[3] = x[1] * e * x[2] * z
        return grad
    end, m)

    return MultiObjectiveProblem(
        n,
        m,
        objectives;
        name = meta.name,
        bounds = (fill(-2.0, n), fill(2.0, n)),
        jacobian = gradients,
    )
end

"""
    MGH16(; nobj::Int = 5)

Return the Brown--Dennis instance with 4 variables and `nobj >= 4` objectives. The
default is `nobj = 5`. The variable bounds are `[-25, 25]`, `[-5, 5]`, `[-5, 5]`,
and `[-1, 1]`. An analytical Jacobian is registered; Hessians are not.
"""
function MGH16(; nobj::Int = 5)
    nobj >= 4 || throw(ArgumentError("nobj must be at least 4 for MGH16"))
    m = nobj
    meta = META["MGH16"]
    n = default_nvar(meta)

    tdata = ntuple(i -> i / 5, m)

    objectives = ntuple(i -> function (x::AbstractVector{T}) where {T <: AbstractFloat}
        ti = T(tdata[i])
        a = x[1] + ti * x[2] - exp(ti)
        b = x[3] + x[4] * sin(ti) - cos(ti)
        return a^2 + b^2
    end, m)

    gradients = ntuple(i -> function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        ti = T(tdata[i])
        s = sin(ti)
        a = x[1] + ti * x[2] - exp(ti)
        b = x[3] + x[4] * s - cos(ti)
        grad[1] = T(2) * a
        grad[2] = T(2) * ti * a
        grad[3] = T(2) * b
        grad[4] = T(2) * s * b
        return grad
    end, m)

    return MultiObjectiveProblem(
        n,
        m,
        objectives;
        name = meta.name,
        bounds = ([-25.0, -5.0, -5.0, -1.0], [25.0, 5.0, 5.0, 1.0]),
        jacobian = gradients,
    )
end

"""
    MGH26(; nvar::Int = 4)

Return the trigonometric instance with `nvar >= 1` variables and `nobj = nvar`
objectives. The default is `nvar = 4`, and all variables are bounded by `[-1, 1]`. An analytical
Jacobian is registered; Hessians are not.
"""
function MGH26(; nvar::Int = 4)
    nvar >= 1 || throw(ArgumentError("nvar must be at least 1 for MGH26"))
    n = nvar
    nobj = nvar
    m = nobj
    meta = META["MGH26"]

    objectives = Vector{Function}(undef, m)
    for i in 1:m
        objectives[i] = function (x::AbstractVector{T}) where {T <: AbstractFloat}
            sum_cos = zero(T)
            @inbounds for k in 1:n
                sum_cos += cos(x[k])
            end
            h = T(n) - sum_cos + T(i) * (one(T) - cos(x[i])) - sin(x[i])
            return h^2
        end
    end

    gradients = Vector{Function}(undef, m)
    for i in 1:m
        gradients[i] = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
            sum_cos = zero(T)
            @inbounds for k in 1:n
                sum_cos += cos(x[k])
            end

            h = T(n) - sum_cos + T(i) * (one(T) - cos(x[i])) - sin(x[i])
            scale = T(2) * h

            @inbounds for k in 1:n
                grad[k] = scale * sin(x[k])
            end
            grad[i] += scale * (T(i) * sin(x[i]) - cos(x[i]))
            return grad
        end
    end

    return MultiObjectiveProblem(
        n,
        m,
        objectives;
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = gradients,
    )
end

"""
    MGH33(; nvar::Int = 10, nobj::Int = 10)

Return the linear rank-1 instance with independent dimensions `nvar >= 2` and
`nobj >= 2`. Both default to 10, and all variables are bounded by `[-1, 1]`. An
analytical Jacobian is registered; Hessians are not.
"""
function MGH33(; nvar::Int = 10, nobj::Int = 10)
    nvar >= 2 || throw(ArgumentError("nvar must be at least 2 for MGH33"))
    nobj >= 2 || throw(ArgumentError("nobj must be at least 2 for MGH33"))
    n = nvar
    m = nobj
    meta = META["MGH33"]

    objectives = ntuple(i -> function (x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for j in 1:n
            s += T(j) * x[j]
        end
        h = T(i) * s - one(T)
        return h^2
    end, m)

    gradients = ntuple(i -> function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        s = zero(T)
        @inbounds for j in 1:n
            s += T(j) * x[j]
        end

        h = T(i) * s - one(T)
        scale = T(2) * h * T(i)
        @inbounds for j in 1:n
            grad[j] = scale * T(j)
        end
        return grad
    end, m)

    return MultiObjectiveProblem(
        n,
        m,
        objectives;
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = gradients,
    )
end
