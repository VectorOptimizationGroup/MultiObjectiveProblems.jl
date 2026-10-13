# MultiObjectiveProblems.jl

[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://vectoroptimizationgroup.github.io/MultiObjectiveProblems.jl/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://vectoroptimizationgroup.github.io/MultiObjectiveProblems.jl/dev/)
[![CI](https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/VectorOptimizationGroup/MultiObjectiveProblems.jl/graph/badge.svg?branch=main)](https://codecov.io/gh/VectorOptimizationGroup/MultiObjectiveProblems.jl)

MultiObjectiveProblems.jl is a curated Julia library of benchmark problems for
multiobjective optimization. It provides source-traceable problem
implementations, a consistent evaluation API, registered analytical
derivatives, and metadata-based catalog queries.

## Installation

```julia
import Pkg
Pkg.add("MultiObjectiveProblems")
```

See the [installation guide](https://vectoroptimizationgroup.github.io/MultiObjectiveProblems.jl/stable/installation/)
for the development version, project environments, and local checkouts.

## Quick Example

```julia
using MultiObjectiveProblems
using Random

prob = DTLZ2()
lower, upper = recommended_bounds(prob)
rng = MersenneTwister(1234)
α = rand(rng, prob.nvar)
x = lower .+ α .* (upper .- lower)

values = eval_f(prob, x)
J = eval_jacobian(prob, x)

names = filter_problems(has_jacobian = true)
```

## Documentation

The [MultiObjectiveProblems.jl documentation](https://vectoroptimizationgroup.github.io/MultiObjectiveProblems.jl/dev/)
contains the quick start, task-oriented guides, mathematical formulations,
API reference, and bibliography. Instructions for building it locally are in
[`docs/README.md`](docs/README.md).

## Highlights

- Curated, source-traceable benchmark implementations.
- Allocating and in-place APIs for objectives and registered derivatives.
- Catalog filters for dimensions, bounds, constraints, and derivative support.
- Registered variable bounds, and a documented finite box for the problems
  whose formulation declares none.
- Per-family formulations, metadata, usage examples, and references.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development workflow, testing requirements, and source-evidence checklist.

## Citation

If you use MultiObjectiveProblems.jl in research, please cite the software using the
metadata in [`CITATION.cff`](CITATION.cff).

## License

MultiObjectiveProblems.jl is distributed under the [MIT License](LICENSE).
