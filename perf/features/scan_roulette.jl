include(joinpath(@__DIR__, "common.jl"))
perf_setup = () -> make_state(:scan)
perf_workload = state -> run_trace(state, :roulette)
perf_oracle = state -> trace_oracle(state, :roulette)
