using PerfChecker
using BenchmarkTools
using Statistics
using SHA

# A bounded PerfChecker campaign using a shared-process BenchmarkTools executor.
# Run only after source changes have stopped. Compilation is excluded from trials.
# timeout 900s julia --project=perf/controller perf/workspace_small.jl OUTPUT.csv
length(ARGS) == 1 || error("usage: workspace_small.jl OUTPUT.csv")
const WORKSPACE_SMALL_OUTPUT = abspath(only(ARGS))
const WORKSPACE_SMALL_ROOT = dirname(@__DIR__)
push!(LOAD_PATH, WORKSPACE_SMALL_ROOT)
include(joinpath(@__DIR__, "workspace_small_common.jl"))
const WORKSPACE_SMALL_STARTED = time()
const WORKSPACE_SMALL_MAX_SECONDS = 900.0
const WORKSPACE_SMALL_MAX_RSS = 2 << 30
const WORKSPACE_SMALL_CACHE = Ref{Any}(nothing)
const WORKSPACE_SMALL_METADATA = Dict{Tuple{Int,Symbol},Any}()
const WORKSPACE_SMALL_ROWS = NamedTuple[]

function workspace_small_fingerprint()
    paths = sort(vcat([joinpath(WORKSPACE_SMALL_ROOT, "Project.toml"), @__FILE__,
                      joinpath(@__DIR__, "workspace_small_common.jl")],
        [joinpath(root, file) for (root, _, files) in walkdir(joinpath(WORKSPACE_SMALL_ROOT, "src"))
                             for file in files if endswith(file, ".jl")]))
    return bytes2hex(sha256(join((relpath(p, WORKSPACE_SMALL_ROOT)*":"*
                                 bytes2hex(sha256(read(p))) for p in paths), "\n")))
end
const WORKSPACE_SMALL_FINGERPRINT = workspace_small_fingerprint()

function workspace_small_cached(n, family)
    key = (n, family)
    cached = WORKSPACE_SMALL_CACHE[]
    cached !== nothing && cached.key == key && return cached
    setup = @timed workspace_small_fixture(n, family)
    setup.bytes <= WORKSPACE_SMALL_LIMITS.max_fixture_alloc_bytes || error("fixture allocation budget exceeded")
    state = setup.value
    plan = workspace_small_artifact(state, Val(:prepared))
    workspace = ProductWorkspace(plan, state.inputs[1]...;
                                 max_bytes=WORKSPACE_SMALL_LIMITS.max_workspace_bytes)
    metadata = (; support=state.support, paths=length(plan.paths), slots=length(plan.output_masks),
                  plan_bytes=Base.summarysize(plan), workspace_bytes=Base.summarysize(workspace),
                  fixture_ms=setup.time*1000, fixture_bytes=setup.bytes)
    WORKSPACE_SMALL_METADATA[key] = metadata
    cached = (; key, state, plan, workspace, metadata)
    WORKSPACE_SMALL_CACHE[] = cached
    return cached
end

function workspace_small_executor(planned, config, setup, workload)
    time() - WORKSPACE_SMALL_STARTED <= WORKSPACE_SMALL_MAX_SECONDS || error("campaign wall-time budget exhausted")
    Sys.maxrss() <= WORKSPACE_SMALL_MAX_RSS || error("campaign RSS budget exhausted")
    case = planned.feature.options[:workspace_case]
    (; n, family, horizon, strategy, phase) = case
    cached = workspace_small_cached(n, family)
    (; state, plan, workspace) = cached
    trial = if phase == :prepare
        preparation = Val(strategy)
        warm = @timed workspace_small_prepare(state, plan, preparation)
        warm.bytes <= WORKSPACE_SMALL_LIMITS.max_batch_alloc_bytes || error("preparation allocation budget exceeded")
        check_strategy = strategy == :plan_build ? Val(:prepared) : Val(:workspace)
        workspace_small_oracle(state, check_strategy, warm.value) || error("preparation exact oracle failed")
        @benchmark workspace_small_prepare($state, $plan, $preparation) samples=12 evals=1 seconds=0.1
    else
        operation = Val(strategy)
        mode = Val(phase)
        artifact = strategy == :direct ? nothing : strategy == :prepared ? plan : workspace
        workspace_small_oracle(state, operation, artifact) || error("full-output exact oracle failed")
        expected = workspace_small_expected_checksum(state, horizon)
        warm = @timed workspace_small_batch(state, operation, horizon, mode, artifact)
        warm.value == expected || error("horizon exact checksum failed")
        warm.bytes <= WORKSPACE_SMALL_LIMITS.max_batch_alloc_bytes || error("horizon allocation budget exceeded")
        measured = @benchmark workspace_small_batch($state, $operation, $horizon, $mode, $artifact) samples=12 evals=1 seconds=0.1
        workspace_small_batch(state, operation, horizon, mode, artifact) == expected ||
            error("post-sampling horizon checksum failed")
        workspace_small_oracle(state, operation, artifact) || error("post-sampling full-output oracle failed")
        measured
    end
    trial.memory <= WORKSPACE_SMALL_LIMITS.max_batch_alloc_bytes || error("measured allocation budget exceeded")
    metadata = cached.metadata
    row = (; case..., status=:pass, median_us=median(trial.times)/1000,
             p95_us=quantile(trial.times, 0.95)/1000, memory_bytes=trial.memory,
             allocs=trial.allocs, samples=length(trial.times), metadata..., reason="")
    push!(WORKSPACE_SMALL_ROWS, row)
    workspace_small_append_row(WORKSPACE_SMALL_OUTPUT*".partial", row)
    qualification = Dict{String,Any}(
        "correctness" => Dict("status" => "passed", "required" => true,
            "message" => "independent Int64 inversion oracle, full outputs, changed coefficients, executed horizon checksum"),
        "source_fingerprint" => WORKSPACE_SMALL_FINGERPRINT,
        "execution" => Dict("mode" => "shared_process_screening"))
    return PerfChecker.CheckerResult([PerfChecker.to_table(trial)], nothing,
        [:garamon, :workspace_small], [PerfChecker.PackageSpec(name="Garamon")], [qualification])
end

function workspace_small_append_row(path, row)
    contract = row.phase == :prepare ? "preparation_only" :
        row.strategy == :values ? "borrowed_coefficient_vector" : "consumed_full_multivector"
    open(path, "a") do io
        println(io, join((row.n, row.family, row.horizon, row.strategy, row.phase, contract,
            row.status, round(row.median_us; digits=3), round(row.p95_us; digits=3),
            row.memory_bytes, row.allocs, row.samples, row.support, row.paths, row.slots,
            row.plan_bytes, row.workspace_bytes, round(row.fixture_ms; digits=3),
            row.fixture_bytes, row.paths*row.horizon, row.reason), ","))
    end
end

function workspace_small_write_amortization(path)
    open(path, "w") do io
        println(io, "# source_sha256=$WORKSPACE_SMALL_FINGERPRINT; completion and source-stability qualification are in the main CSV")
        println(io, "# model uses measured preparation and per-operation medians; observed horizon uses actual end-to-end loops")
        println(io, "dimension,family,horizon,strategy,contract,prepare_us,direct_batch_us,steady_batch_us,end_to_end_batch_us,modeled_break_even_calls,first_observed_winning_horizon")
        records = Dict((r.n,r.family,r.horizon,r.strategy,r.phase)=>r for r in WORKSPACE_SMALL_ROWS)
        for family in WORKSPACE_SMALL_FAMILIES, n in 2:12, strategy in (:prepared,:workspace,:values)
            winning = Int[]
            for horizon in WORKSPACE_SMALL_HORIZONS
                direct = get(records,(n,family,horizon,:direct,:steady),nothing)
                inclusive = get(records,(n,family,horizon,strategy,:end_to_end),nothing)
                direct === nothing || inclusive === nothing ||
                    inclusive.median_us >= direct.median_us || push!(winning,horizon)
            end
            for horizon in WORKSPACE_SMALL_HORIZONS
                direct = get(records,(n,family,horizon,:direct,:steady),nothing)
                steady = get(records,(n,family,horizon,strategy,:steady),nothing)
                inclusive = get(records,(n,family,horizon,strategy,:end_to_end),nothing)
                preparation = get(records,(n,family,0,strategy == :prepared ? :plan_build : :plan_workspace_build,:prepare),nothing)
                any(isnothing,(direct,steady,inclusive,preparation)) && continue
                benefit = (direct.median_us - steady.median_us)/horizon
                model = benefit > 0 ? string(max(1,ceil(Int,preparation.median_us/benefit))) : "none"
                # The vector interface has a different output contract. Retain its raw
                # timings above, but do not claim a full-multivector break-even point.
                contract = strategy == :values ? "different_contract" : "consumed_full_multivector"
                first_win = strategy == :values ? "not_comparable" : isempty(winning) ? "none_measured" : string(minimum(winning))
                strategy == :values && (model="not_comparable")
                println(io,join((n,family,horizon,strategy,contract,
                    round(preparation.median_us;digits=3),round(direct.median_us;digits=3),
                    round(steady.median_us;digits=3),round(inclusive.median_us;digits=3),model,first_win),","))
            end
        end
    end
end

mktempdir() do entrypoint_directory
    cases = NamedTuple[]
    skips = NamedTuple[]
    for family in WORKSPACE_SMALL_FAMILIES, n in 2:12
        if workspace_small_admission(n,family) == "admitted"
            for strategy in (:plan_build,:workspace_build,:plan_workspace_build)
                push!(cases,(;n,family,horizon=0,strategy,phase=:prepare))
            end
        end
        for horizon in WORKSPACE_SMALL_HORIZONS, strategy in WORKSPACE_SMALL_STRATEGIES, phase in (:steady,:end_to_end)
            case = (;n,family,horizon,strategy,phase)
            reason = workspace_small_admission(n,family,horizon)
            if reason == "admitted"
                push!(cases,case)
            else
                push!(skips,(;case...,reason))
            end
        end
    end
    common = joinpath(@__DIR__,"workspace_small_common.jl")
    entrypoint = joinpath(entrypoint_directory,"workspace_small.jl")
    # Required FeatureSpec source identity. The custom executor owns fixture and
    # measurement dispatch; this entrypoint is not an isolated-worker workload.
    write(entrypoint,"include("*repr(common)*")\n")
    features = [FeatureSpec(Symbol(:workspace_,case.n,:_,case.family,:_,case.horizon,:_,case.strategy,:_,case.phase);
        description="Workspace $(case.family), n=$(case.n), H=$(case.horizon), $(case.phase), $(case.strategy)",
        backend=:benchmark,entrypoint,oracle=OracleSpec(),
        comparison_key="garamon/workspace/$(case.family)/$(case.n)d/H$(case.horizon)/$(case.phase)/$(case.strategy == :values ? "vector" : "multivector")/v1",
        options=Dict(:workspace_case=>case,:samples=>12,:evals=>1,:seconds=>0.1)) for case in cases]
    package = PackageSuite("Garamon";worker_environment=joinpath(@__DIR__,"runner"),
        source=WORKSPACE_SMALL_ROOT,versions=VersionNumber[],dev_sources=String[],features)
    suite = SoftwareSuite(:garamon_workspace_small,[package];description="Real reuse horizons with exact full-output oracles")
    mkpath(dirname(WORKSPACE_SMALL_OUTPUT))
    partial = WORKSPACE_SMALL_OUTPUT*".partial"
    open(partial,"w") do io
        println(io,"# Julia=$VERSION, cpu=$(Sys.CPU_NAME), threads=$(Threads.nthreads()), mode=shared_process_screening, evals=1, samples_max=12, seconds_target=0.1")
        println(io,"# source_sha256=$WORKSPACE_SMALL_FINGERPRINT, limits=$WORKSPACE_SMALL_LIMITS, max_campaign_seconds=$WORKSPACE_SMALL_MAX_SECONDS, max_rss_bytes=$WORKSPACE_SMALL_MAX_RSS")
        println(io,"dimension,family,horizon,strategy,phase,contract,status,median_us,p95_us,memory_bytes,allocs,samples,support,paths,output_slots,plan_bytes,workspace_bytes,fixture_ms,fixture_bytes,batch_paths,reason")
    end
    reported = Ref(-1)
    function progress(payload)
        completed = payload["completed"]
        if completed != reported[] && (completed % 32 == 0 || completed == length(cases))
            reported[] = completed
            println("Workspace progress $completed/$(length(cases)); passed=$(payload["passed"]), failed=$(payload["failed"]), elapsed=",round(time()-WORKSPACE_SMALL_STARTED;digits=1)," s")
            flush(stdout)
        end
    end
    println("Workspace campaign: $(length(cases)) admitted cases, $(length(skips)) explicit skips")
    result = run_suite(suite;profile=:quick,strict=false,executor=workspace_small_executor,progress_callback=progress)
    unchanged = workspace_small_fingerprint() == WORKSPACE_SMALL_FINGERPRINT
    open(partial,"a") do io
        for (run,case) in zip(result.runs,cases)
            run.status == :pass && continue
            reason = replace(run.message,','=>';','\n'=>' ')
            println(io,join((case.n,case.family,case.horizon,case.strategy,case.phase,"",run.status,
                "","","","","","","","","","","","","",reason),","))
        end
        for case in skips
            println(io,join((case.n,case.family,case.horizon,case.strategy,case.phase,"",:skipped,
                "","","","","","","","","","","","","",case.reason),","))
        end
        println(io,"# completed=true, source_unchanged=$unchanged, elapsed_seconds=",time()-WORKSPACE_SMALL_STARTED,", maxrss_bytes=",Sys.maxrss())
    end
    mv(partial,WORKSPACE_SMALL_OUTPUT;force=true)
    workspace_small_write_amortization(splitext(WORKSPACE_SMALL_OUTPUT)[1]*"-amortization.csv")
    println("Workspace verdict: ",suite_verdict(result),"; source unchanged: ",unchanged,"; CSV: ",WORKSPACE_SMALL_OUTPUT)
    unchanged || error("source changed during measurements")
    suite_passed(result) || error("workspace campaign has failed cases")
end
