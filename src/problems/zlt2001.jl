"""
    ZLT1(; nvar::Int = 100, nobj::Int = 2)

Construct the `ZLT1` problem with configurable `nvar` and `nobj`.

Requires `nobj >= 2` and `nvar >= nobj`. Each variable is bounded in
`[-1000, 1000]`. An analytical Jacobian is registered; objective Hessians are
not registered.
"""
function ZLT1(; nvar::Int = 100, nobj::Int = 2)
    nobj >= 2 || throw(ArgumentError("nobj must be at least 2 for ZLT1"))
    nvar >= nobj || throw(ArgumentError("nvar must be at least nobj for ZLT1"))
    n = nvar
    m = nobj
    meta = META["ZLT1"]

    objectives = ntuple(m) do idx
        function (x::AbstractVector{T}) where {T <: AbstractFloat}
            s = (x[idx] - one(T))^2
            @inbounds for i in 1:n
                if i != idx
                    s += x[i]^2
                end
            end
            return s
        end
    end

    gradients = ntuple(m) do idx
        function (grad::AbstractVector{T}, x::AbstractVector{T}) where {T <: AbstractFloat}
            @inbounds for i in 1:n
                grad[i] = T(2) * x[i]
            end
            # Overwrite the shifted coordinate, whose derivative is 2 * (x[idx] - 1).
            grad[idx] = T(2) * (x[idx] - one(T))
            return grad
        end
    end

    return MultiObjectiveProblem(
        n, m, objectives;
        name = meta.name,
        bounds = (fill(-1000.0, n), fill(1000.0, n)),
        jacobian = gradients,
    )
end
