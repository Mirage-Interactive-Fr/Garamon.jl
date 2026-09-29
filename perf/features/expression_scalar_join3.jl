include(joinpath(@__DIR__, "common_expression_grades.jl"))
perf_setup = expression_grade_state
perf_workload = state -> expression_grade_workload(state, :join3)
perf_oracle = state -> expression_grade_oracle(state, :join3)
