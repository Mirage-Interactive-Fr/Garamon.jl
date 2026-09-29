using TOML

length(ARGS) == 4 || error("worker arguments: scenario strategy repetition output")
scenario, strategy = Symbol(ARGS[1]), Symbol(ARGS[2])
repetition = parse(Int, ARGS[3])
destination = ARGS[4]

function rss(field="VmRSS")
    m = match(Regex("(?m)^" * field * ":\\s+(\\d+)\\s+kB"), read("/proc/self/status", String))
    m === nothing ? -1 : 1024parse(Int, m.captures[1])
end
function observe(f, args...)
    @nospecialize f args
    @timed Base.invokelatest(f, args...)
end
observe(identity, nothing)
const LOAD_RSS_BEFORE = rss()
const LOAD_JIT_BEFORE = Base.jit_total_bytes()
loading = @timed @eval using GaramonLifecycle
const LOAD_FINISHED_SINCE_LAUNCH_MS = 1000(time() - parse(Float64, ENV["LIFECYCLE_LAUNCH_UNIX"]))
const LOAD_JIT_DELTA = Base.jit_total_bytes() - LOAD_JIT_BEFORE
const LOAD_RSS_AFTER = rss()
const L = GaramonLifecycle
L.BLAS.set_num_threads(1)

function row(stage, index, sample, horizon, observation, jit, before, after)
    Dict("scenario" => String(scenario), "strategy" => String(strategy),
        "repetition" => repetition, "stage" => stage, "context" => index,
        "sample" => sample, "horizon" => horizon, "time_ms" => 1000observation.time,
        "compile_ms" => 1000observation.compile_time,
        "recompile_ms" => 1000observation.recompile_time,
        "allocated_bytes" => observation.bytes, "gc_ms" => 1000observation.gctime,
        "jit_added_bytes" => jit, "rss_before_bytes" => before,
        "rss_after_bytes" => after)
end

function measure!(rows, stage, index, sample, horizon, f, args...)
    @nospecialize f args
    before = rss()
    jit_before = Base.jit_total_bytes()
    observation = observe(f, args...)
    jit = Base.jit_total_bytes() - jit_before
    after = rss()
    push!(rows, row(stage, index, sample, horizon, observation, jit, before, after))
    observation.value
end

function campaign()
    rows = Dict[]
    push!(rows, row("load", 0, 0, 0, loading, LOAD_JIT_DELTA, LOAD_RSS_BEFORE, LOAD_RSS_AFTER))
    fixtures, artifacts = Any[], Any[]
    plan_types, topologies = Set{String}(), Set{Any}()
    first_result_since_launch_ms = NaN
    indices = scenario == :known ? (0:0) : (1:12)
    # These functions are not invoked before the recorded first occurrence.
    for index in indices
        fixture = measure!(rows, "discovery", index, 0, 0, L.lifecycle_fixture, index)
        plan = measure!(rows, "plan_first", index, 0, 0, L.lifecycle_plan, fixture)
        push!(plan_types, string(typeof(plan)))
        push!(topologies, Tuple((p[1], p[2], p[3]) for p in plan.paths))
        artifact = measure!(rows, "artifact_first", index, 0, 0,
                            L.lifecycle_artifact, plan, fixture, strategy)
        result = measure!(rows, "execute_first", index, 0, 1,
                          L.lifecycle_execute, artifact, fixture, strategy)
        isempty(fixtures) && (first_result_since_launch_ms =
            1000(time() - parse(Float64, ENV["LIFECYCLE_LAUNCH_UNIX"])))
        expected = L.lifecycle_reference(fixture, 0)
        actual = zeros(Float64, 16)
        for (mask, coefficient) in result.values
            actual[Int(mask)+1] = coefficient
        end
        actual == expected || error("first-execution oracle failed")
        push!(fixtures, fixture)
        push!(artifacts, artifact)
    end
    # Warm structural reconstruction, separated from execution. Five independent
    # plans/artifacts are created; no lookup cache substitutes an existing object.
    fixture = first(fixtures)
    for sample in 1:5
        plan = measure!(rows, "plan_rebuild_hot", first(indices), sample, 0,
                        L.lifecycle_plan, fixture)
        measure!(rows, "artifact_rebuild_hot", first(indices), sample, 0,
                 L.lifecycle_artifact, plan, fixture, strategy)
    end
    # Warm the trace harness separately, preserving its first compilation cost.
    output = measure!(rows, "trace_first", -1, 0, 32,
                      L.lifecycle_trace, fixtures, artifacts, strategy, 32)
    L.lifecycle_validate(fixtures, output)
    for horizon in (1, 32, 1024), sample in 1:5
        output = measure!(rows, "execute_hot", -1, sample, horizon,
                          L.lifecycle_trace, fixtures, artifacts, strategy, horizon)
        L.lifecycle_validate(fixtures, output)
    end
    GC.gc()
    metadata = Dict("retained_fixture_artifact_bytes" => Base.summarysize((fixtures, artifacts)),
                    "rss_after_gc_bytes" => rss(), "peak_rss_bytes" => rss("VmHWM"),
                    "contexts" => length(fixtures), "plan_types" => length(plan_types),
                    "topologies" => length(topologies), "oracle" => "pass",
                    "launch_to_load_finished_ms" => LOAD_FINISHED_SINCE_LAUNCH_MS,
                    "launch_to_first_result_ms" => first_result_since_launch_ms)
    open(destination, "w") do io
        TOML.print(io, Dict("rows" => rows, "metadata" => metadata))
    end
end
campaign()
