# Installation

The installation commands below are illustrative: they modify the active
environment or use a placeholder path.

## Install the registered release

MultiObjectiveProblems.jl is registered in the Julia General registry. Add it to
the active Julia environment by name:

```julia
import Pkg
Pkg.add("MultiObjectiveProblems")
```

Using a project-specific environment is recommended so that the package
version is recorded alongside the rest of the project's dependencies. See the
Julia `Pkg` documentation for environment creation and activation.

## Install the development version

To use changes that are not yet in a registered release, install the `main`
branch directly from GitHub:

```julia
import Pkg
Pkg.add(url = "https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl")
```

## Install from a local checkout

Clone the repository when you need a local copy of the source:

```bash
git clone https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl.git
```

To install that checkout into the active Julia environment, provide its local
path:

```julia
import Pkg
Pkg.add(path = "/absolute/path/to/MultiObjectiveProblems.jl")
```

Use `Pkg.develop(path = ...)` instead when local source edits should be visible
immediately from the active environment.

## Verify the installation

```jldoctest installation_verify
julia> using MultiObjectiveProblems

julia> using Random

julia> prob = ZDT1();

julia> lower, upper = recommended_bounds(prob);

julia> rng = MersenneTwister(1234);

julia> α = rand(rng, prob.nvar);

julia> x = lower .+ α .* (upper .- lower);

julia> values = eval_f(prob, x);

julia> length(values)
2
```

Continue with the [Quick Start](@ref) for the main construction, evaluation,
and catalog workflow.
