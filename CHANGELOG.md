# Changelog

Notable user-visible changes to MultiObjectiveProblems.jl are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Analytical Jacobians for `AAS1` and `AAS2`, so every problem in the catalog
  now registers one; `filter_problems(has_jacobian = true)` includes both.

### Changed

- Installation instructions use the registered package,
  `Pkg.add("MultiObjectiveProblems")`; installing from GitHub is documented as
  the development version.

## [0.1.0] - 2026-10-12

### Added

- A curated catalog of 73 multiobjective benchmark problems with consistent
  allocating and in-place evaluation APIs.
- Registered analytical Jacobians, selected analytical Hessians, constraint
  evaluation, dimension metadata, catalog filters, and recommended finite
  working boxes.
- Source-traceable family documentation and executable examples.
- Exported problem type `MultiObjectiveProblem`, returned by every benchmark
  constructor, and metadata type `ProblemMeta`.

[Unreleased]: https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl/releases/tag/v0.1.0
