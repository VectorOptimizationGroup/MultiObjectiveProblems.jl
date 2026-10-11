# Changelog

Notable user-visible changes to MultiObjectiveProblems.jl are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A curated catalog of 73 multiobjective benchmark problems with consistent
  allocating and in-place evaluation APIs.
- Registered analytical Jacobians, selected analytical Hessians, constraint
  evaluation, dimension metadata, catalog filters, and recommended finite
  working boxes.
- Source-traceable family documentation and executable examples.
- Exported problem type `MultiObjectiveProblem`, returned by every benchmark
  constructor, and metadata type `ProblemMeta`.

[Unreleased]: https://github.com/VectorOptimizationGroup/MultiObjectiveProblems.jl/commits/main
