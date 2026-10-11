using Test
using LinearAlgebra
using FiniteDiff
using MultiObjectiveProblems
using .TestUtils

# A 3ⁿ grid at the quartiles of the box from `recommended_bounds` when n ≤ 2;
# one sampled point otherwise.
function grid_points(prob::MultiObjectiveProblem)
    prob.nvar <= 2 || return [TestUtils.sample_x(prob)]
    l, u = recommended_bounds(prob)
    levels = [[Float64(l[i]) + t * (Float64(u[i]) - Float64(l[i])) for t in (0.25, 0.5, 0.75)]
              for i in 1:prob.nvar]
    return [collect(p) for p in Iterators.product(levels...)]
end

@testset "Hessian: analytic vs FD" begin
    names = MultiObjectiveProblems.filter_problems(has_hessian=true)
    for name in names
        meta = MultiObjectiveProblems.META[name]
        variable = !(meta.dimension isa FixedDimension)
        Ns = variable ? TestUtils.dims() : ()

        if variable
            for n in Ns
                @testset "$(name) n=$(n)" begin
                    local prob
                    try
                        prob = TestUtils.instantiate_with_dimension(name, n)
                    catch e
                        @error "Failed to instantiate" name=name n=n error=e
                        @test false
                        continue
                    end
                    if isnothing(prob.hessian)
                        continue
                    end
                    for (k, x) in enumerate(grid_points(prob))
                        for i in 1:prob.nobj
                            fi = y -> MultiObjectiveProblems.eval_f(prob, y, i)
                            Hx = MultiObjectiveProblems.eval_hessian_row(prob, x, i)
                            ok, relerr = TestUtils.check_hessian(fi, Hx, x)
                            @info "Hessian check" name=name n=prob.nvar obj=i point=k relerr=relerr
                            @test ok
                        end
                    end
                end
            end
        else
            @testset "$(name) default" begin
                local prob
                try
                    prob = getfield(MultiObjectiveProblems, Symbol(name))()
                catch e
                    @error "Failed to instantiate" name=name error=e
                    @test false
                    continue
                end
                if isnothing(prob.hessian)
                    continue
                end
                for (k, x) in enumerate(grid_points(prob))
                    for i in 1:prob.nobj
                        fi = y -> MultiObjectiveProblems.eval_f(prob, y, i)
                        Hx = MultiObjectiveProblems.eval_hessian_row(prob, x, i)
                        ok, relerr = TestUtils.check_hessian(fi, Hx, x)
                        @info "Hessian check" name=name n=prob.nvar obj=i point=k relerr=relerr
                        @test ok
                    end
                end
            end
        end
    end
end
