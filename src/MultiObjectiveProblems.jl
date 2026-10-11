module MultiObjectiveProblems

using LinearAlgebra

include("types.jl")

include("evaluation.jl")
include("catalog.jl")

# Each metadata file defines `<basename>_meta::ProblemMeta`.
for file in filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, "Meta")))
    include(joinpath("Meta", file))
    meta_sym = Symbol(replace(file, ".jl" => "_meta"))
    meta = getfield(@__MODULE__, meta_sym)
    META[meta.name] = meta
end

# Constructors access `META`, so problem files must be included after metadata.
for file in filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, "problems")))
    include(joinpath("problems", file))
end

# Problem and metadata types
export MultiObjectiveProblem, ProblemMeta

# Evaluation API
export eval_f, eval_f!
export eval_c, eval_c!
export eval_jacobian, eval_jacobian!, eval_jacobian_row, eval_jacobian_row!
export eval_hessian, eval_hessian!, eval_hessian_row, eval_hessian_row!
export eval_constraint_jacobian, eval_constraint_jacobian!
export eval_constraint_jacobian_row, eval_constraint_jacobian_row!
export eval_constraint_hessian, eval_constraint_hessian!
export eval_constraint_hessian_row, eval_constraint_hessian_row!

# Catalog and metadata API
export META
export get_problem_names, filter_problems, recommended_bounds
export AbstractDimensionSpec, FixedDimension, VariableNvar, VariableNobj
export IndependentDimension, ParametricDimension, CoupledDimension
export default_nvar, default_nobj

# Benchmark constructors
export AAS1, AAS2
export AP1, AP2, AP3, AP4
export BK1
export DD1
export DGO0, DGO1, DGO2
export DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5
export FA1
export Far1
export FDS
export FF1
export Hil1
export IKK1
export IM1
export JOS1, JOS4
export KW2
export LE1
export Lov1, Lov2, Lov3, Lov4, Lov5, Lov6
export LTDZ, LTDZ1
export MGH9, MGH16, MGH26, MGH33
export MHHM1, MHHM2
export MLF1, MLF2
export MMR1, MMR2, MMR3, MMR4
export MOP2, MOP3, MOP5, MOP6, MOP7
export PNR
export QV1
export SD
export SK1, SK2
export SLCDT1, SLCDT2
export SP1
export SSFYY2
export TKLY1
export Toi4, Toi8, Toi9, Toi10
export VU1, VU2
export ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
export ZLT1

end # module MultiObjectiveProblems
