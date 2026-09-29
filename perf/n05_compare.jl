using SHA, TOML
if abspath(PROGRAM_FILE)==@__FILE__
    using PerfChecker
end
include("n05_pfaffian.jl")

"""Declared measurement condition, not an automatic isolation certificate."""
function n05_condition(environment=ENV)
    explicit=haskey(environment,"GARAMONBENCH_CONDITION")
    condition=strip(get(environment,"GARAMONBENCH_CONDITION","exploratory_interference"))
    condition in ("exploratory_interference","isolated") ||
        throw(ArgumentError("GARAMONBENCH_CONDITION must be exploratory_interference or isolated"))
    label=strip(get(environment,"GARAMONBENCH_INTERFERENCE_LABEL",explicit ? "" : "Etendue3D"))
    condition=="exploratory_interference" && isempty(label) &&
        throw(ArgumentError("exploratory_interference requires GARAMONBENCH_INTERFERENCE_LABEL"))
    condition=="isolated" && !isempty(label) &&
        throw(ArgumentError("isolated requires an absent or empty GARAMONBENCH_INTERFERENCE_LABEL"))
    any(iscntrl,label) && throw(ArgumentError("interference label must be a single printable line"))
    length(label)<=256 || throw(ArgumentError("interference label exceeds 256 characters"))
    source=explicit ? "environment" : "legacy_direct_default"
    jobs=condition=="isolated" ? "none declared; isolation not automatically verified" : "$label; exploratory only"
    (;condition,label,source,concurrent_external_jobs=jobs)
end

function n05_source_hash()
    root=dirname(@__DIR__)
    files=sort(vcat(filter(f->endswith(f,".jl"),readdir(joinpath(root,"src");join=true)),
        [joinpath(@__DIR__,name) for name in ("n05_pfaffian.jl","n05_compare.jl","n05_shards.jl")]))
    bytes2hex(sha256(join((basename(f)*"\n"*read(f,String) for f in files),"\n")))
end

include("n05_shards.jl")

function n05_setup(n,k,horizon,family,strategy,trace)
    BLAS.set_num_threads(1)
    state=n05_fixture(n,k,horizon,family;precontracted=strategy==:precontracted)
    first=@timed Base.invokelatest(n05_episode,state,strategy)
    Sys.maxrss()<=2<<30 || error("N05 worker RSS checkpoint budget exceeded")
    n05_qualify(state,first.value) || error("N05 numerical qualification failed")
    errors=[abs(BigFloat(first.value[t])-BigFloat(state.exact[1+mod(t-1,3)])) for t in eachindex(first.value)]
    if !isfile(trace)
        details=Dict("n"=>n,"chain_length"=>k,"horizon"=>horizon,"metric"=>string(family),
            "strategy"=>string(strategy),"first_episode_ms"=>1000first.time,
            "compile_ms"=>1000first.compile_time,"first_episode_bytes"=>first.bytes,
            "max_absolute_error"=>string(maximum(errors;init=big(0.))),
            "oracle_values"=>string.(state.exact),"construction_included"=>strategy!=:precontracted,
            "preexisting_inputs"=>strategy==:precontracted ? "contractions" : "metric_and_vector_coordinates",
            "qualified"=>true,"fixture_bytes"=>Base.summarysize(state),"worker_peak_rss_bytes"=>Sys.maxrss())
        open(io->TOML.print(io,details),trace,"w")
    end
    state
end

function n05_precision_cases()
    rows=Dict{String,Any}[]
    for exponent in (10,30,50)
        delta=2.0^-exponent;G=[1. 0.;0. -1.]
        V=[1. 1. 1. 1.;1. 1-delta 1. 1-delta]
        exact=get(n05_oracle(G,V),big(0),big(0)//1)
        value=n05_scalar(G,V)
        error=abs(BigFloat(value)-BigFloat(exact))
        push!(rows,Dict("delta_exponent"=>exponent,"exact"=>string(exact),"float64"=>value,
            "absolute_error"=>string(error),"relative_error"=>iszero(exact) ? "undefined" : string(error/abs(BigFloat(exact))),
            "rational_pfaffian_equal"=>n05_scalar(Rational{BigInt}.(G),Rational{BigInt}.(V))==exact))
    end
    rows
end

function n05_campaign(output;smoke=false,prepare_only=false,reverse_order=false)
    started=time()
    condition=n05_condition() # Validate before creating any result files.
    mkpath(output)
    matrix=n05_grid()
    open(joinpath(output,"n05-replay-matrix.csv"),"w") do io
        println(io,"n,chain_length,horizon,metric,strategy,threads,construction_included")
        for r in matrix
            println(io,join((r.n,r.k,r.horizon,r.family,r.strategy,1,r.strategy!=:precontracted),','))
        end
    end
    prepare_only && return println("N05 prepared ",length(matrix)," identities; no timing")
    isfile(joinpath(output,"n05-samples.csv")) && error("choose a fresh result directory")
    # This bounded screening set is declared before any timings.
    fixtures=smoke ? ((2,4,1,:signed),(65,8,32,:signed),(4,6,1,:general),(8,4,32,:null)) :
        ((2,4,1,:euclidean),(8,8,32,:signed),(65,16,32,:null),(129,8,1024,:signed),
         (4,8,32,:general),(8,16,32,:zero))
    cases=[(;n,k,horizon,family,strategy) for (n,k,horizon,family) in fixtures
        for strategy in (:full,:recursive,:pfaffian,:precontracted)
        if strategy!=:recursive || family!=:general]
    reverse_order && reverse!(cases)
    root=dirname(@__DIR__);fingerprint=n05_source_hash();diagnostics=Dict{String,Any}[]
    mktempdir(;prefix="garamon-n05-") do temporary
        features=FeatureSpec[];traces=String[]
        for (i,r) in enumerate(cases)
            entry=joinpath(temporary,"case-$i.jl");trace=joinpath(temporary,"trace-$i.toml");push!(traces,trace)
            open(entry,"w") do io
                println(io,"include(",repr(@__FILE__),")")
                println(io,"perf_setup()=n05_setup(",r.n,",",r.k,",",r.horizon,",:",r.family,",:",r.strategy,",",repr(trace),")")
                println(io,"perf_workload(state)=n05_episode(state,:",r.strategy,")")
                println(io,"perf_oracle(state)=n05_qualify(state,perf_workload(state))")
            end
            contract=r.strategy==:precontracted ? "preexisting_contractions" : "coordinates_construction_included"
            push!(features,FeatureSpec(Symbol(:n05_,i);backend=:benchmark,entrypoint=entry,
                description="N05 ordered vector-chain scalar; $contract; $(r.strategy)",
                comparison_key="n05/$contract/$(r.n)/$(r.k)/$(r.horizon)/$(r.family)",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>0.1,:threads=>1)))
        end
        sum(filesize,filter(isfile,readdir(temporary;join=true));init=0)<=64<<20 || error("N05 temporary disk budget")
        package=PackageSuite("Garamon";source=root,worker_environment=joinpath(@__DIR__,"runner"),
            versions=VersionNumber[],dev_sources=String[],features)
        result=run_suite(SoftwareSuite(:n05_pfaffian,[package]);profile=:quick,strict=false)
        write_suite_json(result,joinpath(output,"n05-qualification.json"))
        open(joinpath(output,"n05-samples.csv"),"w") do io
            println(io,"n,chain_length,horizon,metric,strategy,construction_included,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations")
            for (run,r) in zip(result.runs,cases)
                if run.status==:pass
                    table=only(run.result.tables)
                    for j in eachindex(table.times)
                        println(io,join((r.n,r.k,r.horizon,r.family,r.strategy,r.strategy!=:precontracted,run.status,j,
                            table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((r.n,r.k,r.horizon,r.family,r.strategy,r.strategy!=:precontracted,run.status,"","","","",""),','))
                end
            end
        end
        for trace in traces;isfile(trace) && push!(diagnostics,TOML.parsefile(trace));end
        open(io->TOML.print(io,Dict("episodes"=>diagnostics,"near_cancellation"=>n05_precision_cases())),joinpath(output,"n05-diagnostics.toml"),"w")
        suite_passed(result) || error("N05 qualification failure; raw results preserved")
    end
    fingerprint==n05_source_hash() || error("N05 measured sources changed")
    elapsed=time()-started
    elapsed<=900 || error("N05 wall-time checkpoint budget exceeded")
    details=Dict("source_sha256"=>fingerprint,"julia"=>string(VERSION),"perfchecker"=>string(pkgversion(PerfChecker)),
        "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"cpu"=>Sys.cpu_info()[1].model,
        "measurement_condition"=>condition.condition,"condition_declaration"=>condition.source,
        "interference_label"=>condition.label,"concurrent_external_jobs"=>condition.concurrent_external_jobs,
        "reverse_order"=>reverse_order,
        "cases"=>length(cases),"elapsed_seconds"=>elapsed,"controller_peak_rss_bytes"=>Sys.maxrss(),
        "fixture_budget_bytes"=>64<<20,"worker_rss_checkpoint_budget_bytes"=>2<<30,
        "wall_budget_seconds"=>900,"temporary_and_results_budget_bytes"=>64<<20,
        "oracle"=>"independent rational Chevalley exterior action",
        "float_atol"=>1e-8,"float_rtol"=>1e-10,"native_cache_between_sessions_tested"=>false,
        "controller_manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"controller","Manifest.toml")))))
    open(io->TOML.print(io,details),joinpath(output,"n05-environment.toml"),"w")
    sum(filesize,filter(f->startswith(basename(f),"n05-")&&isfile(f),readdir(output;join=true));init=0)<=64<<20 || error("N05 result disk budget")
    println("N05: ",length(cases)," qualified cases; signed and degenerate metrics; one thread")
end

if abspath(PROGRAM_FILE)==@__FILE__
    flags=filter(arg->startswith(arg,"--"),ARGS)
    selection=n05_parse_selection(flags)
    paths=filter(arg->!startswith(arg,"--"),ARGS)
    length(paths)==1 || error("usage: n05_compare.jl [--smoke | --shard=INDEX/COUNT | --range=FIRST:LAST] [--prepare-only] [--reverse] OUTPUT")
    if selection.shard!==nothing || selection.interval!==nothing
        n05_shard_campaign(abspath(only(paths));selection...,prepare_only="--prepare-only" in flags,reverse_order="--reverse" in flags)
    else
        n05_campaign(abspath(only(paths));smoke="--smoke" in flags,prepare_only="--prepare-only" in flags,reverse_order="--reverse" in flags)
    end
end
