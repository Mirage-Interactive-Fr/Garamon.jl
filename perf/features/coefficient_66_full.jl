include(joinpath(@__DIR__, "common_coefficient.jl"))
perf_setup = () -> product_state(66)
perf_workload = state -> coefficient_workload(state, :full)
perf_oracle = state -> coefficient_oracle(state, :full)
