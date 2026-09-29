using PerfChecker
using BenchmarkTools
using Statistics
using SHA

# Contiguous, bounded screening. The default executor shares one Julia process
# across cases; --isolated uses PerfChecker's standard per-feature workers.
# External hard timeout example: timeout 900s julia --project=perf/controller \
#   perf/explore_dimensions.jl /tmp/garamon-dimensions.csv
# Setup/oracle are excluded from timings. There is one request batch per sample.
length(ARGS) >= 1 || error("usage: explore_dimensions.jl OUTPUT.csv [MAX_DIMENSION=256] [--isolated] [--dimensions=63,64,65]")
const output_path = abspath(ARGS[1])
const max_dimension = length(ARGS) >= 2 && !startswith(ARGS[2], "--") ? parse(Int, ARGS[2]) : 256
2 <= max_dimension <= 256 || error("MAX_DIMENSION must be between 2 and 256")
const isolated = "--isolated" in ARGS
const dimension_flag = findfirst(arg -> startswith(arg, "--dimensions="), ARGS)
const dimensions = dimension_flag === nothing ? collect(2:max_dimension) :
    sort!(unique(parse.(Int, split(last(split(ARGS[dimension_flag], "="; limit=2)), ","))))
!isempty(dimensions) && all(n -> 2 <= n <= max_dimension, dimensions) ||
    error("selected dimensions must be nonempty and within 2:MAX_DIMENSION")
const strategies = (:full, :recursive, :recursive_grades, :join3)
const requests = (:scalar, :four)
const common = joinpath(@__DIR__, "features", "common_expression_dimensions.jl")
push!(LOAD_PATH, dirname(@__DIR__))
include(common)

# Fingerprint before and after: concurrent source edits invalidate a campaign.
function campaign_fingerprint()
    paths = sort(vcat([joinpath(dirname(@__DIR__), "Project.toml"), @__FILE__, common],
        [joinpath(root, f) for (root, _, files) in walkdir(joinpath(dirname(@__DIR__), "src"))
                          for f in files if endswith(f, ".jl")]))
    return bytes2hex(sha256(join((relpath(p, dirname(@__DIR__)) * ":" *
                                 bytes2hex(sha256(read(p))) for p in paths), "\n")))
end
const initial_fingerprint = campaign_fingerprint()
const state_cache = Dict{Tuple{Int,Symbol,Symbol},Any}()
const setup_metrics = Dict{Tuple{Int,Symbol,Symbol},Any}()
const started_at = time()
const campaign_seconds = 900.0
const campaign_maxrss_bytes = 2 << 30

function screening_executor(planned, config, setup, workload)
    time() - started_at < campaign_seconds || error("campaign cooperative wall-time budget exhausted")
    Sys.maxrss() <= campaign_maxrss_bytes || error("campaign cooperative peak-RSS budget exhausted")
    options = planned.feature.options
    n, shape, request, strategy = options[:dimension_case]
    key = (n, shape, request)
    if !haskey(state_cache, key)
        empty!(state_cache) # retain only one state, not the whole dimension sweep
        setup_result = @timed expression_dimension_state(n, shape, request)
        setup_result.bytes <= setup_result.value.limits.max_setup_bytes ||
            error("family setup allocation budget exceeded")
        state_cache[key] = setup_result.value
        setup_metrics[key] = (; seconds=setup_result.time, bytes=setup_result.bytes,
                              support=setup_result.value.reference_support,
                              pairs=setup_result.value.oracle_pairs)
    end
    state = state_cache[key]
    expression_dimension_oracle(state, strategy) || error("independent exact integer oracle failed")
    # A warm execution bounds observed Julia allocations before sampling.
    warm = @timed expression_dimension_workload(state, strategy)
    warm.bytes <= state.limits.max_alloc_bytes || error("family allocation budget exceeded")
    trial = @benchmark expression_dimension_workload($state, $strategy) samples=12 evals=1 seconds=0.05
    trial.memory <= state.limits.max_alloc_bytes || error("family allocation budget exceeded")
    expression_dimension_oracle(state, strategy) || error("post-sampling exact oracle failed")
    qualification = Dict{String,Any}(
        "correctness" => Dict("status" => "passed", "required" => true,
                              "message" => "independent Int64 basis-inversion oracle; Float64 exact range"),
        "execution" => Dict("mode" => "shared_process_screening", "setup_excluded" => true),
        "source_fingerprint" => initial_fingerprint)
    return PerfChecker.CheckerResult([PerfChecker.to_table(trial)], nothing,
        [:garamon, :dimension_screen], [PerfChecker.PackageSpec(name="Garamon")], [qualification])
end

mktempdir() do entrypoint_directory
    admitted = NamedTuple[]
    skipped = NamedTuple[]
    features = FeatureSpec[]
    # Same family/dimension/request stays adjacent so all strategies share its fixture.
    for shape in EXPRESSION_DIMENSION_FAMILIES, n in dimensions, request in requests, strategy in strategies
        case = (; n, shape, request, strategy)
        reason = expression_dimension_admission(n, shape)
        if reason != "admitted"
            push!(skipped, (; case..., reason))
            continue
        end
        push!(admitted, case)
        name = Symbol(:expression_, n, :_, shape, :_, request, :_, strategy)
        entrypoint = joinpath(entrypoint_directory, "$(name).jl")
        open(entrypoint, "w") do io
            println(io, "include(", repr(common), ")")
            println(io, "perf_setup = () -> expression_dimension_state($n, :$shape, :$request)")
            println(io, "perf_workload = state -> expression_dimension_workload(state, :$strategy)")
            println(io, "perf_oracle = state -> expression_dimension_oracle(state, :$strategy)")
        end
        push!(features, FeatureSpec(name;
            description="Dimension $n, family $shape, output $request, strategy $strategy",
            backend=:benchmark, entrypoint,
            comparison_key="garamon/expression/$shape/$(n)d/$request/v2",
            oracle=OracleSpec(function_name=:perf_oracle),
            options=Dict(:tags => [:garamon, :expression, shape, request, strategy],
                         :dimension_case => (n, shape, request, strategy),
                         :samples => 12, :evals => 1, :seconds => 0.05)))
    end
    package = PackageSuite("Garamon";
        worker_environment=joinpath(@__DIR__, "runner"), source=dirname(@__DIR__),
        versions=VersionNumber[], dev_sources=String[], features)
    suite = SoftwareSuite(:garamon_dimension_screen, [package];
        description="Every integer dimension, independent exact oracle, bounded workload families")
    println("Starting $(length(admitted)) cases; $(length(skipped)) explicit budget skips; mode=",
            isolated ? "isolated" : "shared_process_screening")
    result = isolated ? run_suite(suite; profile=:quick, strict=false) :
        run_suite(suite; profile=:quick, strict=false, executor=screening_executor)
    unchanged = campaign_fingerprint() == initial_fingerprint
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, "# Julia=$VERSION, samples_max=12, evals=1, seconds_target=0.05, prep=excluded, oracle=exact_integer, mode=",
                isolated ? "isolated" : "shared_process_screening")
        println(io, "# source_sha256=$initial_fingerprint, source_unchanged=$unchanged, elapsed_seconds=", time()-started_at,
                ", maxrss_bytes=", Sys.maxrss(), ", cooperative_campaign_seconds=$campaign_seconds, cooperative_maxrss_bytes=$campaign_maxrss_bytes")
        for shape in EXPRESSION_DIMENSION_FAMILIES
            println(io, "# family=$shape, ", expression_dimension_limits(shape))
        end
        println(io, "dimension,shape,request,strategy,support_per_operand,request_count,status,median_us,p95_us,memory_bytes,allocs,samples,setup_ms,setup_bytes,reference_support,oracle_pairs,reason")
        for (run, case) in zip(result.runs, admitted)
            (; n, shape, request, strategy) = case
            metadata = get(setup_metrics, (n, shape, request), nothing)
            prep = metadata === nothing ? ("", "", "", "") :
                (round(metadata.seconds*1000; digits=3), metadata.bytes, metadata.support, metadata.pairs)
            if run.status == :pass && run.result isa PerfChecker.CheckerResult
                sample = only(run.result.tables)
                println(io, join((n, shape, request, strategy, expression_dimension_support(n, shape),
                    request == :scalar ? 1 : 4, unchanged ? :pass : :source_changed,
                    round(median(sample.times)/1000; digits=3),
                    round(quantile(sample.times, 0.95)/1000; digits=3),
                    Int(round(median(sample.memory))), Int(round(median(sample.allocs))),
                    length(sample.times), prep..., ""), ","))
            else
                reason = replace(run.message, ',' => ';', '\n' => ' ')
                println(io, join((n, shape, request, strategy, expression_dimension_support(n, shape),
                                 request == :scalar ? 1 : 4, run.status, "", "", "", "", "", prep..., reason), ","))
            end
        end
        for case in skipped
            println(io, join((case.n, case.shape, case.request, case.strategy, "", "", :skipped,
                             "", "", "", "", "", "", "", "", "", case.reason), ","))
        end
    end
    println("PerfChecker verdict: ", suite_verdict(result), "; cases: ", length(result.runs),
            "; source unchanged: ", unchanged, "; elapsed seconds: ", round(time()-started_at; digits=2),
            "; peak RSS bytes: ", Sys.maxrss(), "; CSV: ", output_path)
    unchanged || error("source changed during screening; repeat before using measurements")
    suite_passed(result) || error("dimension exploration has failed cases")
end
