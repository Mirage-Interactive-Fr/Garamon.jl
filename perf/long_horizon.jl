using PerfChecker
using Statistics
using SHA

const LONG_SMOKE = length(ARGS) == 2 && first(ARGS) == "--smoke"
(length(ARGS) == 1 || LONG_SMOKE) || error("usage: julia --startup-file=no --project=perf/controller perf/long_horizon.jl [--smoke] OUTPUT_DIRECTORY")
const LONG_OUTPUT = abspath(last(ARGS))
const LONG_FAMILIES = LONG_SMOKE ? ((66, :stable),) :
    ((66, :stable), (66, :varying), (66, :phases), (128, :phases))
const LONG_HORIZONS = LONG_SMOKE ? (32,) : (1, 32, 1024, 10000)
const LONG_STRATEGIES = LONG_SMOKE ? (:generated, :workspace) :
    (:direct, :prepared, :generated, :recursive, :join3, :workspace)

function long_controller_source_hash()
    directory = joinpath(@__DIR__, "..", "src")
    files = sort!(filter(path -> endswith(path, ".jl"), readdir(directory; join=true)))
    bytes2hex(sha256(join((basename(path) * "\n" * read(path, String) for path in files), "\n")))
end

function long_campaign()
    mkpath(LONG_OUTPUT)
    common = joinpath(@__DIR__, "long_horizon_common.jl")
    fingerprint = long_controller_source_hash()
    mktempdir() do temporary
        features = FeatureSpec[]
        cases = [(n, trace, horizon, strategy) for (n, trace) in LONG_FAMILIES
                 for horizon in LONG_HORIZONS for strategy in LONG_STRATEGIES]
        cold_paths = String[]
        for (n, trace, horizon, strategy) in cases
            name = Symbol(:long_, n, :_, trace, :_, horizon, :_, strategy)
            cold = joinpath(temporary, "$name.csv")
            push!(cold_paths, cold)
            entrypoint = joinpath(temporary, "$name.jl")
            open(entrypoint, "w") do io
                println(io, "include(", repr(common), ")")
                println(io, "perf_setup = () -> long_setup_with_cold(", n, ", :", trace,
                        ", ", horizon, ", :", strategy, ", ", repr(cold), ")")
                println(io, "perf_workload = state -> long_trace(state, :", strategy, ")")
                println(io, "perf_oracle = state -> long_oracle(state, :", strategy, ")")
            end
            push!(features, FeatureSpec(name;
                description="Actual $horizon triple-product scalar outputs in $n D; $trace; $strategy",
                backend=:benchmark, entrypoint,
                comparison_key="garamon/long/$(n)d/$trace/$horizon/v1",
                oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:tags => [:garamon, :long_horizon, trace, strategy],
                             :samples => 5, :evals => 1, :seconds => 0.2)))
        end
        package = PackageSuite("Garamon";
            worker_environment=joinpath(@__DIR__, "runner"), source=dirname(@__DIR__),
            versions=VersionNumber[], dev_sources=String[], features)
        suite = SoftwareSuite(:garamon_long_horizon, [package];
            description="Actual long-lived high-dimensional product chains with changing coefficients")
        result = run_suite(suite; profile=:quick, strict=false)
        open(joinpath(LONG_OUTPUT, "warm.csv"), "w") do io
            println(io, "dimension,trace,horizon,strategy,status,total_median_ms,total_p95_ms,outputs_per_second,allocated_bytes,allocs,samples")
            for (run, (n, trace, horizon, strategy)) in zip(result.runs, cases)
                if run.status == :pass && run.result isa PerfChecker.CheckerResult
                    sample = only(run.result.tables)
                    median_ns = median(sample.times)
                    println(io, join((n, trace, horizon, strategy, run.status,
                        median_ns/1e6, quantile(sample.times, .95)/1e6,
                        horizon/(median_ns/1e9), Int(round(median(sample.memory))),
                        Int(round(median(sample.allocs))), length(sample.times)), ','))
                else
                    println(io, join((n, trace, horizon, strategy, run.status,
                                      "", "", "", "", "", ""), ','))
                end
            end
        end
        first_cold = true
        open(joinpath(LONG_OUTPUT, "cold.csv"), "w") do io
            for path in cold_paths
                isfile(path) || continue
                lines = readlines(path)
                foreach(line -> println(io, line), first_cold ? lines : lines[2:end])
                first_cold = false
            end
        end
        open(joinpath(LONG_OUTPUT, "environment.txt"), "w") do io
            println(io, "Julia: ", VERSION, "\nPerfChecker: ", pkgversion(PerfChecker),
                    "\nCPU: ", Sys.cpu_info()[1].model,
                    "\nSource: ", dirname(@__DIR__),
                    "\nSource SHA256: ", fingerprint,
                    "\nCommon SHA256: ", bytes2hex(sha256(read(common))),
                    "\nController SHA256: ", bytes2hex(sha256(read(@__FILE__))),
                    "\nPerfChecker verdict: ", suite_verdict(result),
                    "\nFeature count: ", length(result.runs))
        end
        println("Long-horizon PerfChecker verdict: ", suite_verdict(result),
                "; cases: ", length(result.runs), "; output: ", LONG_OUTPUT)
        fingerprint == long_controller_source_hash() || error("source changed during campaign")
        suite_passed(result) || error("long-horizon campaign contains failed cases")
    end
end

long_campaign()
