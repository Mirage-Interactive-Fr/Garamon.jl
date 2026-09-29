include(joinpath(@__DIR__, "common_product.jl"))
perf_setup = () -> product_state(4)
perf_workload = state -> product_workload(state, :direct)
perf_oracle = state -> product_oracle(state, :direct)
