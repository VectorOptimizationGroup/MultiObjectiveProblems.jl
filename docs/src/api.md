# API Reference

This reference presents the docstrings of the types and functions shared by
the benchmark problems, including their signatures, arguments, and behavior.
For workflows that connect these APIs, see [Evaluation and Derivatives](@ref)
and [Catalog and Metadata](@ref).

## Evaluation

General constraints use the representation ``l_c \leq c(x) \leq u_c``, stored
in `prob.lcon` and `prob.ucon`. Rows with equal lower and upper bounds are
equalities; all other rows are inequalities. Objective and constraint
derivatives have separate evaluation functions.

Derivative metadata records whether an analytical evaluator is registered; it
does not assert differentiability at every boundary point of a benchmark's
domain. Family pages document problem-specific restrictions, and registered
evaluators may throw a `DomainError` where an analytical derivative is
undefined.

```@docs
MultiObjectiveProblems.eval_f
MultiObjectiveProblems.eval_f!
MultiObjectiveProblems.eval_c
MultiObjectiveProblems.eval_c!
MultiObjectiveProblems.eval_jacobian
MultiObjectiveProblems.eval_jacobian!
MultiObjectiveProblems.eval_jacobian_row
MultiObjectiveProblems.eval_jacobian_row!
MultiObjectiveProblems.eval_constraint_jacobian
MultiObjectiveProblems.eval_constraint_jacobian!
MultiObjectiveProblems.eval_constraint_jacobian_row
MultiObjectiveProblems.eval_constraint_jacobian_row!
MultiObjectiveProblems.eval_hessian
MultiObjectiveProblems.eval_hessian!
MultiObjectiveProblems.eval_hessian_row
MultiObjectiveProblems.eval_hessian_row!
MultiObjectiveProblems.eval_constraint_hessian
MultiObjectiveProblems.eval_constraint_hessian!
MultiObjectiveProblems.eval_constraint_hessian_row
MultiObjectiveProblems.eval_constraint_hessian_row!
```

## Catalog

For a worked example of selecting candidates, interpreting catalog defaults,
and constructing an instance, see [Catalog and Metadata](@ref).

```@docs
MultiObjectiveProblems.META
MultiObjectiveProblems.get_problem_names
MultiObjectiveProblems.filter_problems
MultiObjectiveProblems.recommended_bounds
```

## Core Types

```@docs
MultiObjectiveProblems.MultiObjectiveProblem
MultiObjectiveProblems.ProblemMeta
MultiObjectiveProblems.AbstractDimensionSpec
MultiObjectiveProblems.FixedDimension
MultiObjectiveProblems.VariableNvar
MultiObjectiveProblems.VariableNobj
MultiObjectiveProblems.IndependentDimension
MultiObjectiveProblems.ParametricDimension
MultiObjectiveProblems.CoupledDimension
MultiObjectiveProblems.default_nvar
MultiObjectiveProblems.default_nobj
```
