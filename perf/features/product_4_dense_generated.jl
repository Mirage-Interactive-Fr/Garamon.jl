include(joinpath(@__DIR__, "common_product.jl"))
perf_setup = () -> product_state(4; storage=:dense)
perf_workload = state -> product_workload(state, :generated)
perf_oracle = state -> product_oracle(state, :generated)
