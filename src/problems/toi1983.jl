# The names retain the numbering of the partially separable test problems of
# Toint (1983). Each objective is one element function of the corresponding
# single-objective problem, following the multiobjective adaptations of Mita,
# Fukuda, and Yamashita (2019), whose bounds the constructors adopt.

"""
    Toi4()

Construct the fixed four-variable, two-objective `Toi4` problem.

The variables are bounded in `[-2, 5]^4`. An analytical Jacobian is registered;
objective Hessians are not registered.
"""
function Toi4()
    meta = META["Toi4"]
    n = default_nvar(meta)
    m = default_nobj(meta)

    f1 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return x[1]^2 + x[2]^2 + one(T)
    end

    f2 = function (x::AbstractVector{T}) where {T <: AbstractFloat}
        return T(0.5) * ((x[1] - x[2])^2 + (x[3] - x[4])^2) + one(T)
    end

    df1_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        grad[1] = T(2) * x[1]
        grad[2] = T(2) * x[2]
        grad[3] = zero(T)
        grad[4] = zero(T)
        return grad
    end

    df2_dx = function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
        diff12 = x[1] - x[2]
        diff34 = x[3] - x[4]
        grad[1] = diff12
        grad[2] = -diff12
        grad[3] = diff34
        grad[4] = -diff34
        return grad
    end

    return MultiObjectiveProblem(
        n, m, (f1, f2);
        name = meta.name,
        bounds = (fill(-2.0, n), fill(5.0, n)),
        jacobian = (df1_dx, df2_dx),
    )
end

"""
    Toi8(; nvar::Int = 3)

Construct the `Toi8` problem with `nobj = nvar`.

Requires `nvar >= 2`. Each variable is bounded in `[-1, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function Toi8(; nvar::Int = 3)
    nvar >= 2 || throw(ArgumentError("nvar must be at least 2 for Toi8"))
    n = nvar
    nobj = nvar
    m = nobj
    meta = META["Toi8"]

    f_list = Vector{Function}(undef, m)
    for idx in 1:m
        f_list[idx] = let idx = idx
            if idx == 1
                function (x::AbstractVector{T}) where {T <: AbstractFloat}
                    return (T(2) * x[1] - one(T))^2
                end
            else
                function (x::AbstractVector{T}) where {T <: AbstractFloat}
                    return T(idx) * (T(2) * x[idx - 1] - x[idx])^2
                end
            end
        end
    end

    jac_rows = Vector{Function}(undef, m)
    for idx in 1:m
        jac_rows[idx] = let idx = idx
            if idx == 1
                function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                    fill!(grad, zero(T))
                    grad[1] = T(4) * (T(2) * x[1] - one(T))
                    return grad
                end
            else
                function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                    fill!(grad, zero(T))
                    diff = T(2) * x[idx - 1] - x[idx]
                    grad[idx - 1] = T(4) * T(idx) * diff
                    grad[idx] = -T(2) * T(idx) * diff
                    return grad
                end
            end
        end
    end

    return MultiObjectiveProblem(
        n, m, f_list;
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = jac_rows,
    )
end

"""
    Toi9(; nvar::Int = 4)

Construct the `Toi9` problem with `nobj = nvar`.

Requires `nvar >= 2`. Each variable is bounded in `[-1, 1]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function Toi9(; nvar::Int = 4)
    nvar >= 2 || throw(ArgumentError("nvar must be at least 2 for Toi9"))
    n = nvar
    nobj = nvar
    m = nobj
    meta = META["Toi9"]

    f_list = Vector{Function}(undef, m)
    for idx in 1:m
        f_list[idx] = let idx = idx
            if idx == 1
                function (x::AbstractVector{T}) where {T <: AbstractFloat}
                    return (T(2) * x[1] - one(T))^2 + x[2]^2
                end
            elseif idx < n
                function (x::AbstractVector{T}) where {T <: AbstractFloat}
                    diff = T(2) * x[idx - 1] - x[idx]
                    return T(idx) * diff^2 - T(idx - 1) * x[idx - 1]^2 + T(idx) * x[idx]^2
                end
            else
                function (x::AbstractVector{T}) where {T <: AbstractFloat}
                    diff = T(2) * x[n - 1] - x[n]
                    return T(n) * diff^2 - T(n - 1) * x[n - 1]^2
                end
            end
        end
    end

    jac_rows = Vector{Function}(undef, m)
    for idx in 1:m
        jac_rows[idx] = let idx = idx
            if idx == 1
                function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                    fill!(grad, zero(T))
                    grad[1] = T(4) * (T(2) * x[1] - one(T))
                    grad[2] = T(2) * x[2]
                    return grad
                end
            elseif idx < n
                function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                    fill!(grad, zero(T))
                    diff = T(2) * x[idx - 1] - x[idx]
                    grad[idx - 1] = T(4) * T(idx) * diff - T(2) * T(idx - 1) * x[idx - 1]
                    grad[idx] = -T(2) * T(idx) * diff + T(2) * T(idx) * x[idx]
                    return grad
                end
            else
                function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                    fill!(grad, zero(T))
                    diff = T(2) * x[n - 1] - x[n]
                    grad[n - 1] = T(4) * T(n) * diff - T(2) * T(n - 1) * x[n - 1]
                    grad[n] = -T(2) * T(n) * diff
                    return grad
                end
            end
        end
    end

    return MultiObjectiveProblem(
        n, m, f_list;
        name = meta.name,
        bounds = (fill(-1.0, n), fill(1.0, n)),
        jacobian = jac_rows,
    )
end

"""
    Toi10(; nvar::Int = 4)

Construct the `Toi10` problem with `nobj = nvar - 1`.

Requires `nvar >= 2`. Each variable is bounded in `[-2, 2]`.
An analytical Jacobian is registered; objective Hessians are not registered.
"""
function Toi10(; nvar::Int = 4)
    nvar >= 2 || throw(ArgumentError("nvar must be at least 2 for Toi10"))
    n = nvar
    meta = META["Toi10"]
    nobj = nvar - 1
    m = nobj

    f_rows = Vector{Function}(undef, m)
    for idx in 1:m
        f_rows[idx] = let idx = idx
            function (x::AbstractVector{T}) where {T <: AbstractFloat}
                diff = x[idx + 1] - x[idx]^2
                return T(100) * diff^2 + (x[idx + 1] - one(T))^2
            end
        end
    end

    jac_rows = Vector{Function}(undef, m)
    for idx in 1:m
        jac_rows[idx] = let idx = idx
            function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
                fill!(grad, zero(T))
                diff = x[idx + 1] - x[idx]^2
                grad[idx] = -T(400) * diff * x[idx]
                grad[idx + 1] = T(200) * diff + T(2) * (x[idx + 1] - one(T))
                return grad
            end
        end
    end

    return MultiObjectiveProblem(
        n, m, f_rows;
        name = meta.name,
        bounds = (fill(-2.0, n), fill(2.0, n)),
        jacobian = jac_rows,
    )
end
