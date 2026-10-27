using Test
using MultiObjectiveProblems
using .TestUtils
using Random

@testset "Jacobian: analytic vs FD" begin
    names = MultiObjectiveProblems.filter_problems(has_jacobian=true)
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
                    if isnothing(prob.jacobian)
                        continue
                    end
                    x = TestUtils.sample_x(prob)
                    f = y -> MultiObjectiveProblems.eval_f(prob, y)
                    J = y -> MultiObjectiveProblems.eval_jacobian(prob, y)
                    ok, err = TestUtils.check_jacobian(f, J, x)
                    @info "Jacobian check" name=name n=prob.nvar relerr=err
                    @test ok
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
                if isnothing(prob.jacobian)
                    continue
                end
                x = TestUtils.sample_x(prob)
                f = y -> MultiObjectiveProblems.eval_f(prob, y)
                J = y -> MultiObjectiveProblems.eval_jacobian(prob, y)
                ok, err = TestUtils.check_jacobian(f, J, x)
                @info "Jacobian check" name=name n=prob.nvar relerr=err
                @test ok
            end
        end
    end
end

@testset "JOS4 Jacobian domain" begin
    prob = JOS4(nvar = 3)
    boundary = [0.0, 0.5, 0.5]

    @test eval_f(prob, boundary) == [0.0, 5.5]
    @test eval_jacobian_row(prob, boundary, 1) == [1.0, 0.0, 0.0]

    error = try
        eval_jacobian_row(prob, boundary, 2)
        nothing
    catch exception
        exception
    end
    @test error isa DomainError
    @test occursin("x₁ > 0", sprint(showerror, error))

    @test_throws DomainError eval_jacobian(prob, boundary)
    @test all(isfinite, eval_jacobian(prob, [eps(Float64), 0.5, 0.5]))
end

@testset "FA1 Jacobian domain" begin
    prob = FA1()
    boundary = [0.0, 0.5, 0.5]

    @test eval_f(prob, boundary) == [0.0, 1.5, 1.5]
    @test eval_jacobian_row(prob, boundary, 1) ≈ [4 / (1 - exp(-4)), 0.0, 0.0]

    for row in (2, 3)
        error = try
            eval_jacobian_row(prob, boundary, row)
            nothing
        catch exception
            exception
        end
        @test error isa DomainError
        @test occursin("x₁ > 0", sprint(showerror, error))
    end

    @test_throws DomainError eval_jacobian(prob, boundary)
    @test all(isfinite, eval_jacobian(prob, [eps(Float64), 0.5, 0.5]))
end

@testset "LE1 bounds and Jacobian domain" begin
    prob = LE1()
    origin = [0.0, 0.0]
    second_center = [0.5, 0.5]

    @test prob.bounds == (fill(-5.0, 2), fill(10.0, 2))
    @test eval_f(prob, origin) ≈ [0.0, 0.5^0.25]
    @test eval_f(prob, second_center) ≈ [0.5^0.125, 0.0]

    first_error = try
        eval_jacobian_row(prob, origin, 1)
        nothing
    catch exception
        exception
    end
    @test first_error isa DomainError
    @test occursin("(0, 0)", sprint(showerror, first_error))
    @test all(isfinite, eval_jacobian_row(prob, origin, 2))
    @test_throws DomainError eval_jacobian(prob, origin)

    second_error = try
        eval_jacobian_row(prob, second_center, 2)
        nothing
    catch exception
        exception
    end
    @test second_error isa DomainError
    @test occursin("(0.5, 0.5)", sprint(showerror, second_error))
    @test all(isfinite, eval_jacobian_row(prob, second_center, 1))
    @test_throws DomainError eval_jacobian(prob, second_center)

    @test all(isfinite, eval_jacobian(prob, [0.25, 0.25]))
end

@testset "QV1 Jacobian domain" begin
    prob = QV1(nvar = 3)
    first_center = zeros(3)
    second_center = fill(1.5, 3)

    @test eval_f(prob, first_center)[1] == 0.0
    @test all(isfinite, eval_jacobian_row(prob, first_center, 2))

    first_error = try
        eval_jacobian_row(prob, first_center, 1)
        nothing
    catch exception
        exception
    end
    @test first_error isa DomainError
    @test occursin("first objective minimizer", sprint(showerror, first_error))
    @test_throws DomainError eval_jacobian(prob, first_center)

    @test eval_f(prob, second_center)[2] == 0.0
    @test all(isfinite, eval_jacobian_row(prob, second_center, 1))

    second_error = try
        eval_jacobian_row(prob, second_center, 2)
        nothing
    catch exception
        exception
    end
    @test second_error isa DomainError
    @test occursin("second objective minimizer", sprint(showerror, second_error))
    @test_throws DomainError eval_jacobian(prob, second_center)

    @test all(isfinite, eval_jacobian(prob, fill(0.75, 3)))
end

@testset "ZDT Jacobian domain" begin
    for (constructor, n) in ((ZDT1, 4), (ZDT3, 4), (ZDT4, 4))
        prob = constructor(nvar=n)
        boundary = fill(0.5, n)
        boundary[1] = 0.0

        @test all(isfinite, eval_f(prob, boundary))
        @test eval_jacobian_row(prob, boundary, 1) == [1.0; zeros(n - 1)]

        error = try
            eval_jacobian_row(prob, boundary, 2)
            nothing
        catch exception
            exception
        end
        @test error isa DomainError
        @test occursin("x₁ > 0", sprint(showerror, error))

        @test_throws DomainError eval_jacobian(prob, boundary)

        interior = copy(boundary)
        interior[1] = eps(Float64)
        @test all(isfinite, eval_jacobian(prob, interior))
    end

    prob = ZDT6(nvar = 4)
    boundary = zeros(4)
    boundary[1] = 0.5

    @test all(isfinite, eval_f(prob, boundary))
    @test all(isfinite, eval_jacobian_row(prob, boundary, 1))

    error = try
        eval_jacobian_row(prob, boundary, 2)
        nothing
    catch exception
        exception
    end
    @test error isa DomainError
    @test occursin("x₂ + ⋯ + xₙ > 0", sprint(showerror, error))

    @test_throws DomainError eval_jacobian(prob, boundary)

    interior = copy(boundary)
    interior[2] = eps(Float64)
    @test all(isfinite, eval_jacobian(prob, interior))
end

# The second AAS1 objective and both AAS2 objectives have the form
# `(λ/p) Σ |yᵢ|^p` with `y` affine in `x` and `p` slightly above one, so each
# gradient term `sign(yᵢ) |yᵢ|^(p-1)` is still about `±0.95` at `|yᵢ| = 1e-8`
# and is `0` at `yᵢ = 0`. The kinks below make a component of `y` exactly zero in
# `Float64` and in the `BigFloat` reference; the ones marked `all` also in
# `Float32`. A search of the box found no point other than the centre `c₂` that
# zeroes the second component of AAS2's `y` for `f₂`, so that kink is tested at
# the centre.
# A `Float32` kink is checked against the `Float64` Jacobian at the same kink,
# not against the `BigFloat` reference: the reference rounds the constants to
# `Float64`, sees `yᵢ ≈ 1e-8` instead of zero, and so a term near `±0.95`.
@testset "AAS Jacobian at random points and kinks" begin
    rng = Random.MersenneTwister(2025)
    cases = (
        (AAS1(), [
            ([0.0, 0.0], true),     # f₂: y₁ = y₂ = 0
            ([0.8, -1.0], true),    # f₂: y₁ = 0
            ([-2.0, 0.5], true),    # f₂: y₂ = 0
        ]),
        (AAS2(), [
            ([1.5, -1.0], true),    # f₁: y₁ = y₂ = 0
            ([1.75, 0.0], true),    # f₁: y₁ = 0
            ([0.0, -0.6], false),   # f₁: y₂ = 0
            ([-1.2, 0.8], true),    # f₂: y₁ = y₂ = 0
            ([-0.7, -1.0], false),  # f₂: y₁ = 0
            ([-1.7, 2.6], true),    # f₂: y₁ = 0
        ]),
    )
    # Float32 rounding of the analytical evaluation.
    rtol32 = 100 * eps(Float32)

    for (prob, kinks) in cases
        f = y -> MultiObjectiveProblems.eval_f(prob, y)
        J = y -> MultiObjectiveProblems.eval_jacobian(prob, y)
        l, u = prob.bounds
        randoms = [l .+ rand(rng, prob.nvar) .* (u .- l) for _ in 1:200]

        @testset "$(prob.name) Float64" begin
            for x in vcat(randoms, first.(kinks))
                ok, err = TestUtils.check_jacobian(f, J, x)
                ok || @info "Jacobian mismatch" name=prob.name x=x relerr=err
                @test ok
            end
        end

        @testset "$(prob.name) Float32" begin
            for x in (Float32.(r) for r in randoms)
                @test eltype(J(x)) === Float32
                ok, err = TestUtils.check_jacobian(f, J, x; rtol=rtol32)
                ok || @info "Jacobian mismatch" name=prob.name x=x relerr=err
                @test ok
            end
            for (x, all) in kinks
                all || continue
                J32 = J(Float32.(x))
                @test eltype(J32) === Float32
                @test TestUtils.relok(J32, J(x); rtol=rtol32)
            end
        end
    end

    @test eval_jacobian_row(AAS1(), [0.0, 0.0], 2) == [0.0, 0.0]
    @test eval_jacobian_row(AAS2(), [1.5, -1.0], 1) == [0.0, 0.0]
    @test eval_jacobian_row(AAS2(), [-1.2, 0.8], 2) == [0.0, 0.0]
    @test eval_jacobian_row(AAS1(), Float32[0, 0], 2) == Float32[0, 0]
end
