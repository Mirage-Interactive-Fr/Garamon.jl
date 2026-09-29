include(joinpath(@__DIR__, "common.jl"))
perf_setup = () -> make_state(:hot)
perf_workload = state -> run_trace(state, :direct)
perf_oracle = state -> trace_oracle(state, :direct)
