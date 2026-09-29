include(joinpath(@__DIR__, "common.jl"))
perf_setup = () -> make_state(:phase)
perf_workload = state -> run_trace(state, :lru)
perf_oracle = state -> trace_oracle(state, :lru)
