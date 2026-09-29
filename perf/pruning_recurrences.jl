using PerfChecker, BenchmarkTools, Statistics, SHA
include(joinpath(@__DIR__,"pruning_recurrences_common.jl"))

length(ARGS)==1 || error("usage: pruning_recurrences.jl OUTPUT.csv")
const PRUNING_OUTPUT=abspath(only(ARGS))
const PRUNING_START=time()
const PRUNING_HASH=bytes2hex(sha256(join(read.(
    [@__FILE__,joinpath(@__DIR__,"pruning_recurrences_common.jl")],String),"\n")))
const PRUNING_CACHE=Ref{Any}(nothing)
const PRUNING_ROWS=NamedTuple[]
const PRUNING_SKIPS=NamedTuple[]

function pruning_cached(case)
    key=(case.n,case.signature,case.regime,case.recurrence,case.horizon)
    previous=PRUNING_CACHE[]
    previous!==nothing && previous.key==key && return previous
    created=@timed pruning_fixture(key...)
    fixture=created.value
    started=time()
    oracle=pruning_oracle(fixture)
    time()-started<=10 || error("oracle_wall_budget")
    exact_rational=pruning_run(fixture,:exact,Rational{BigInt};record=true)
    exact_rational.history==oracle || error("independent_oracle_disagrees")
    # Algebraic restoration is checked separately from floating-point equality.
    for strategy in (:deferred,:replay)
        result=pruning_run(fixture,strategy,Rational{BigInt})
        result.value==last(oracle) || error("rational_restoration_failed")
    end
    exact_float=pruning_run(fixture,:exact;record=true)
    cached=(;key,fixture,oracle,exact_float,fixture_us=created.time*1e6,
            fixture_bytes=created.bytes,oracle_seconds=time()-started)
    PRUNING_CACHE[]=cached
    return cached
end

function pruning_append(row)
    open(PRUNING_OUTPUT*".partial","a") do io
        println(io,join(values(row),","))
    end
end

function pruning_executor(planned,config,setup,workload)
    time()-PRUNING_START<=900 || error("campaign_wall_budget")
    Sys.maxrss()<=2<<30 || error("campaign_rss_budget")
    case=planned.feature.options[:pruning_case]
    cache=pruning_cached(case);fixture=cache.fixture;method=case.method
    warm=@timed pruning_run(fixture,method;measure_buffers=false)
    warm.bytes<=256<<20 || error("trajectory_allocation_budget")
    run=pruning_run(fixture,method;record=true,clock_parts=true)
    run.buffer_bytes<=64<<20 || error("trajectory_buffer_budget")
    errors=pruning_errors(run,cache.oracle)
    exact_float_equal=isequal(run.value,cache.exact_float.value)
    method in (:exact,:replay) && !exact_float_equal && error("exact_float_contract_failed")
    # Moment estimates use independent trajectories, outside timing samples.
    variance=0.0;bias=0.0;rmse=0.0;seedmax=errors.final_abs
    if method==:roulette
        samples=hcat([pruning_run(fixture,method;seed,measure_buffers=false).value
                      for seed in 1:32]...)
        meanresult=vec(mean(samples;dims=2))
        truth=Float64.(last(cache.oracle))
        variance=sum(var(samples;dims=2,corrected=true))
        bias=maximum(abs.(meanresult.-truth))
        rmse=sqrt(mean([sum(abs2,samples[:,j].-truth) for j in axes(samples,2)]))
        seedmax=maximum(abs.(samples.-truth))
    end
    trial=@benchmark pruning_run($fixture,$method;measure_buffers=false) samples=7 evals=1 seconds=0.03
    checksum=sum(warm.value.value)
    isfinite(checksum) || error("nonfinite_output")
    row=(;dimension=case.n,signature=case.signature,regime=case.regime,
          recurrence=case.recurrence,horizon=case.horizon,strategy=method,status="pass",
          median_us=median(trial.times)/1000,p95_us=quantile(trial.times,0.95)/1000,
          allocation_bytes=trial.memory,allocations=trial.allocs,samples=length(trial.times),
          fixture_us=cache.fixture_us,fixture_allocation_bytes=cache.fixture_bytes,
          oracle_seconds=cache.oracle_seconds,buffer_bytes=run.buffer_bytes,
          active_coefficients=length(fixture.initial),candidate_paths=run.counts.candidates,
          kept_paths=run.counts.kept,omitted_paths=run.counts.omitted,
          correction_paths=run.counts.correction_paths,replay_paths=run.counts.replay_paths,
          corrections=run.counts.corrections,restoration_us_instrumented=run.restoration_us,
          final_abs_error=errors.final_abs,max_abs_error=errors.max_abs,
          final_relative_error=errors.final_rel,residual_l1_max=run.maxresidue,
          exact_float_equal,rational_restoration=method in (:exact,:deferred,:replay),
          seeds=method==:roulette ? 32 : 1,variance_l2=variance,
          empirical_bias_linf=bias,rmse_l2=rmse,seed_max_abs=seedmax,reason="")
    push!(PRUNING_ROWS,row);pruning_append(row)
    return PerfChecker.CheckerResult([PerfChecker.to_table(trial)],nothing,[:pruning_exploration],
        [PerfChecker.PackageSpec(name="Garamon")],
        [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
            "message"=>"independent rational trajectory; approximate error reported separately"),
            "source_fingerprint"=>PRUNING_HASH)])
end

mktempdir() do temporary
    cases=NamedTuple[]
    for n in PRUNING_DIMS,signature in PRUNING_SIGNATURES,regime in PRUNING_REGIMES,
        recurrence in (:affine,:bilinear)
        for horizon in (recurrence==:affine ? (8,256) : (1,4,8)),method in PRUNING_METHODS
            push!(cases,(;n,signature,regime,recurrence,horizon,method))
        end
    end
    entrypoint=joinpath(temporary,"pruning.jl")
    write(entrypoint,"include("*repr(joinpath(@__DIR__,"pruning_recurrences_common.jl"))*")\n")
    features=[FeatureSpec(Symbol("pruning_",i);description=string(case),backend=:benchmark,
        entrypoint,oracle=OracleSpec(),comparison_key="pruning/$(case.n)/$(case.signature)/$(case.regime)/$(case.recurrence)/$(case.horizon)",
        options=Dict(:pruning_case=>case)) for (i,case) in enumerate(cases)]
    suite=SoftwareSuite(:pruning_recurrences,[PackageSuite("Garamon";
        worker_environment=joinpath(@__DIR__,"runner"),source=dirname(@__DIR__),
        versions=VersionNumber[],dev_sources=String[],features)])
    mkpath(dirname(PRUNING_OUTPUT))
    open(PRUNING_OUTPUT*".partial","w") do io
        println(io,"# Julia=$VERSION cpu=$(Sys.CPU_NAME) threads=$(Threads.nthreads()) source_sha256=$PRUNING_HASH")
        println(io,"# independent experimental coefficient kernels; Float64 timings; Rational{BigInt} oracle excluded; setup excluded and separately recorded; tau=$(PRUNING_TAU); correction_period=$(PRUNING_PERIOD); no core source changes")
        println(io,"dimension,signature,regime,recurrence,horizon,strategy,status,median_us,p95_us,allocation_bytes,allocations,samples,fixture_us,fixture_allocation_bytes,oracle_seconds,buffer_bytes,active_coefficients,candidate_paths,kept_paths,omitted_paths,correction_paths,replay_paths,corrections,restoration_us_instrumented,final_abs_error,max_abs_error,final_relative_error,residual_l1_max,exact_float_equal,rational_restoration,seeds,variance_l2,empirical_bias_linf,rmse_l2,seed_max_abs,reason")
    end
    previous_progress=Ref(-1)
    function progress(payload)
        i=payload["completed"]
        if i%50==0 && i!=previous_progress[]
            previous_progress[]=i
            println("Pruning $i/$(length(cases)) passed=$(payload["passed"]) failed=$(payload["failed"]) elapsed=",round(time()-PRUNING_START;digits=1),"s");flush(stdout)
        end
    end
    result=run_suite(suite;profile=:quick,strict=false,executor=pruning_executor,progress_callback=progress)
    current=bytes2hex(sha256(join(read.([@__FILE__,joinpath(@__DIR__,"pruning_recurrences_common.jl")],String),"\n")))
    open(PRUNING_OUTPUT*".partial","a") do io
        for (case,run) in zip(cases,result.runs)
            run.status==:pass && continue
            reason=replace(run.message,','=>';','\n'=>' ')
            println(io,join((case.n,case.signature,case.regime,case.recurrence,case.horizon,case.method,
                run.status,fill("",28)...,reason),","))
        end
        # A quadratic rational recurrence can double denominator bit lengths at
        # each step. Do not attempt the long horizon under this exact-oracle cap.
        for n in PRUNING_DIMS,signature in PRUNING_SIGNATURES,regime in PRUNING_REGIMES,method in PRUNING_METHODS
            println(io,join((n,signature,regime,:bilinear,256,method,:skipped,
                fill("",28)...,"long_bilinear_not_admitted_under_exact_oracle_bit_budget"),","))
        end
        println(io,"# completed=true source_unchanged=$(current==PRUNING_HASH) elapsed_seconds=$(time()-PRUNING_START) maxrss_bytes=$(Sys.maxrss())")
    end
    mv(PRUNING_OUTPUT*".partial",PRUNING_OUTPUT;force=true)
    println("Pruning verdict: ",suite_verdict(result)," hash=",PRUNING_HASH)
    current==PRUNING_HASH || error("sources_changed")
    suite_passed(result) || error("campaign_has_failures")
end
