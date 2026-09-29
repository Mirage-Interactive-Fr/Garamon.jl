include(joinpath(@__DIR__, "common_expression_grades.jl"))
perf_setup = expression_grade_state
perf_workload = state -> expression_grade_workload(state, :recursive_grades)
perf_oracle = state -> expression_grade_oracle(state, :recursive_grades)
