using PerfChecker, BenchmarkTools, Statistics, SHA, TOML
push!(LOAD_PATH,dirname(@__DIR__))
include("binary_rank_compact_cases.jl")

"""Declared measurement condition; this does not certify machine isolation."""
function compact_condition(environment=ENV)
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

function compact_campaign(output;mode=:smoke,shard_dimension=nothing)
    Threads.nthreads()==1 || error("K1 requires one Julia thread")
    condition=compact_condition() # Validate before creating output directories.
    mkpath(output)
    paths=Dict(key=>joinpath(output,"k1-binary-rank-"*key*ext) for (key,ext) in
        (("samples",".csv"),("summary",".csv"),("protocol",".toml"),("qualification",".json")))
    any(isfile,values(paths)) && error("use a fresh output directory")
    sourcefiles=sort(vcat([@__FILE__],[joinpath(@__DIR__,f) for f in
        ("binary_rank.jl","binary_rank_cases.jl","binary_rank_compact.jl","binary_rank_compact_cases.jl")],
        [joinpath(dirname(@__DIR__),"src",f) for f in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(f,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    initial=fingerprint();started=time();fixturecache=Dict{Tuple,Any}();artifacts=Dict{Tuple,Any}()
    started_ids=String[];measured_ids=String[];passed_ids=String[]
    smoke=mode in (:smoke,:decision_smoke)
    decision=mode in (:decision_smoke,:decision_screen,:prepare,:shard)
    methods=decision ? COMPACT_DECISION_METHODS : COMPACT_METHODS
    if mode==:shard
        shard_dimension isa Int && 2<=shard_dimension<=128 || error("shard dimension must be an integer in 2:128")
    end
    dims=smoke ? (8,) : mode in (:screen,:decision_screen) ? COMPACT_SCREEN_DIMS : mode==:shard ? (shard_dimension,) : 2:128
    families=smoke ? (:low_rank_high_grade,) : RANK_FAMILIES
    records=NamedTuple[]
    for (index,n) in enumerate(dims),family in families
        signatures=mode in (:shard,:prepare) ? RANK_SIGNATURES : (RANK_SIGNATURES[mod1(index,3)],)
        for signature in signatures;push!(records,(;n,family,signature));end
    end
    passages=mode in (:screen,:decision_screen) ? (1:2) : (1:1)
    queries=mode in (:shard,:prepare) ? COMPACT_FULL_QUERIES : ((:all,0),)
    cases=[(;record...,horizon,method,query,passage) for passage in passages for record in records
        for horizon in RANK_HORIZONS for query in queries
        for method in (isodd(passage) ? methods : reverse(methods))]
    protocol=Dict("technique"=>"K1_N01_plus_compact_workspace","schema_version"=>2,"grid_version"=>"K1v2","mode"=>string(mode),
        "timing_status"=>condition.condition,"measurement_condition"=>condition.condition,
        "condition_declaration"=>condition.source,"interference_label"=>condition.label,
        "concurrent_external_jobs"=>condition.concurrent_external_jobs,
        "threads"=>1,"blas_threads"=>1,"julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,
        "source_sha256"=>initial,"horizons"=>collect(RANK_HORIZONS),"screen_dimensions"=>collect(COMPACT_SCREEN_DIMS),
        "integer_dimension_sequences"=>Dict(string(f)=>collect(2:128) for f in RANK_FAMILIES),
        "families"=>collect(string.(RANK_FAMILIES)),"signatures"=>collect(string.(RANK_SIGNATURES)),
        "screen_signature_assignment"=>"one signature per dimension, cycling positive/indefinite/degenerate; full replay crosses all three",
        "degenerate_metric"=>"diagonal entries 1 and n zero; all other entries +1; ensures null directions are active in every support family",
        "coefficient_period"=>4,"coefficient_values"=>collect(RANK_COEFFICIENTS),
        "output_contract"=>"H independently owned ambient sparse outputs retained until observation returns; same four cycling input pairs for all methods",
        "episode_contract"=>"fresh plan plus workspace when applicable, H actual products and H owned outputs per observed call; fixture/oracle excluded; compiled code warm",
        "phase_contract"=>"separate build, hot and complete episode samples; primary PerfChecker table is complete episode; no additive cost model used for verdict",
        "workspace_contract"=>"single serial workspace per episode; compact output and input buffers reused; every returned dictionary is owned and independent",
        "compact_contract"=>"no 2^d decoding allocation; only nonzero query-admissible reachable XOR coordinates; complete ambient cocycle",
        "prepared_contract"=>"current exact prepare_product/run_product route; full prepared product then ambient query projection when requested; same H owned outputs",
        "passages"=>collect(passages),"method_order_by_passage"=>Dict(string(p)=>collect(string.(isodd(p) ? methods : reverse(methods))) for p in passages),
        "partition_contract"=>"127 disjoint dimension shards n002 through n128; each 324 cases, all 3 metrics, 3 families, 3 horizons, 3 queries, 4 methods; global indices 1:41148",
        "selected_shard"=>(mode==:shard ? compact_shard_id(shard_dimension) : "multiple"),
        "completed"=>false,"execution_state"=>(mode==:prepare ? "prepared_only" : "running"),
        "shards"=>[Dict("id"=>compact_shard_id(n),"dimension"=>n,"expected_cases"=>COMPACT_SHARD_SIZE,
            "first_global_index"=>(n-2)*COMPACT_SHARD_SIZE+1,"last_global_index"=>(n-1)*COMPACT_SHARD_SIZE) for n in 2:128],
        "max_rank"=>16,"max_pairs"=>65_536,"max_outputs"=>65_536,"max_plan_bytes"=>64<<20,
        "max_plan_and_workspace_bytes"=>64<<20,"rss_limit_bytes"=>2<<30,"wall_limit_seconds"=>900,
        "max_episode_allocation_bytes"=>512<<20,"samples"=>7,"case_count"=>length(cases),
        "cases"=>[Dict("dimension"=>c.n,"family"=>string(c.family),"signature"=>string(c.signature),"horizon"=>c.horizon,
            "method"=>string(c.method),"query"=>string(c.query),"passage"=>c.passage,
            "case_id"=>compact_case_id(c),"execution_id"=>compact_execution_id(c),"shard_id"=>compact_shard_id(c.n),
            "global_index"=>compact_global_index(c)) for c in cases])
    length(unique(compact_execution_id.(cases)))==length(cases) || error("duplicate execution ID")
    if mode==:prepare
        compact_global_index.(cases)==collect(1:41_148) || error("full manifest is not exhaustive")
    elseif mode==:shard
        length(cases)==COMPACT_SHARD_SIZE || error("shard size differs from manifest")
        compact_global_index.(cases)==collect((shard_dimension-2)*COMPACT_SHARD_SIZE+1:(shard_dimension-1)*COMPACT_SHARD_SIZE) || error("shard indices differ from manifest")
    end
    open(io->TOML.print(io,protocol),paths["protocol"],"w")
    mode==:prepare && return nothing
    open(paths["samples"],"w") do io
        println(io,"dimension,family,signature,query,horizon,method,passage,phase,sample,time_ns,gc_ns,memory_bytes,allocations,case_id,execution_id,shard_id,global_index")
    end
    open(paths["summary"],"w") do io
        println(io,"dimension,family,signature,query,horizon,method,passage,status,rank,pairs,span_slots,stored_output_slots,plan_bytes,artifact_bytes,numerical_buffer_bytes,first_build_us,build_median_us,hot_median_us,episode_median_us,episode_measured_us_per_product,episode_allocated_bytes,episode_allocations,reason,case_id,execution_id,shard_id,global_index")
    end
    function append_samples(case,phase,trial)
        open(paths["samples"],"a") do io
            for i in eachindex(trial.times)
                println(io,join((case.n,case.family,case.signature,case.query[1],case.horizon,case.method,case.passage,phase,i,
                    trial.times[i],trial.gctimes[i],trial.memory,trial.allocs,compact_case_id(case),compact_execution_id(case),
                    compact_shard_id(case.n),compact_global_index(case)),','))
            end
        end
    end
    function executor(planned,config,setup,workload)
        time()-started<=900 || error("campaign_wall_budget")
        Sys.maxrss()<=2<<30 || error("campaign_rss_budget")
        case=planned.feature.options[:compact_case];key=(case.n,case.family,case.signature)
        push!(started_ids,compact_execution_id(case))
        fixture=get!(fixturecache,key) do;compact_fixture(key...);end
        artifactkey=(key...,case.method,case.query,case.passage)
        cached=get!(artifacts,artifactkey) do
            firstbuild=@timed compact_artifact(fixture,case.method,case.query)
            artifact=firstbuild.value
            compact_correct(fixture,case.method,artifact,case.query) || error("Int64_oracle_failed")
            build=@benchmark compact_artifact($fixture,$(case.method),$(case.query)) samples=7 evals=1 seconds=0.02
            append_samples(case,:build,build)
            (;artifact,firstbuild_us=firstbuild.time*1e6,build_us=median(build.times)/1000,metadata=compact_metadata(artifact))
        end
        artifact=cached.artifact;method=case.method;query=case.query;horizon=case.horizon
        warm=compact_execute(fixture,method,artifact,query,horizon)
        rank_owned_correct(fixture,warm,query,horizon) || error("hot_episode_oracle_or_ownership_failed")
        hot=@benchmark compact_execute($fixture,$method,$artifact,$query,$horizon) samples=7 evals=1 seconds=0.02
        append_samples(case,:hot,hot)
        warm=compact_episode(fixture,method,query,horizon)
        rank_owned_correct(fixture,warm,query,horizon) || error("complete_episode_oracle_or_ownership_failed")
        episode=@benchmark compact_episode($fixture,$method,$query,$horizon) samples=7 evals=1 seconds=0.02
        append_samples(case,:episode,episode)
        push!(measured_ids,compact_execution_id(case))
        episode.memory<=512<<20 || error("episode_allocation_budget")
        rank_owned_correct(fixture,compact_episode(fixture,method,query,horizon),query,horizon) || error("post_sample_oracle_or_ownership_failed")
        meta=cached.metadata
        open(paths["summary"],"a") do io
            println(io,join((case.n,case.family,case.signature,query[1],horizon,method,case.passage,:pass,
                meta.rank,meta.pairs,meta.span_slots,meta.slots,meta.plan_bytes,meta.artifact_bytes,meta.numerical_buffer_bytes,
                cached.firstbuild_us,cached.build_us,median(hot.times)/1000,median(episode.times)/1000,
                median(episode.times)/1000/horizon,episode.memory,episode.allocs,"",compact_case_id(case),compact_execution_id(case),
                compact_shard_id(case.n),compact_global_index(case)),','))
        end
        push!(passed_ids,compact_execution_id(case))
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,[:K1,Symbol(condition.condition),:complete_episode],
            [PerfChecker.PackageSpec(name="Garamon")],[Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"independent Int64 oracle; all H ambient outputs, checksum and nonaliasing verified before and after timing"),
                "source_fingerprint"=>initial,"measurement_condition"=>condition.condition,
                "condition_declaration"=>condition.source,"interference_label"=>condition.label,
                "concurrent_external_jobs"=>condition.concurrent_external_jobs,
                "execution"=>Dict("mode"=>"shared_process","threads"=>1,"measurement_condition"=>condition.condition,
                    "interference_label"=>condition.label))])
    end
    mktempdir(;prefix="compact-rank-perfchecker-") do temporary
        entry=joinpath(temporary,"compact.jl");write(entry,"# K1 shared-process PerfChecker executor entrypoint\n")
        features=[FeatureSpec(Symbol(:compact_,i);entrypoint=entry,backend=:benchmark,oracle=OracleSpec(),
            comparison_key="K1/owned/$(c.n)/$(c.family)/$(c.signature)/$(c.query)/$(c.horizon)/pass$(c.passage)",
            options=Dict(:compact_case=>c)) for (i,c) in enumerate(cases)]
        suite=SoftwareSuite(:binary_rank_compact,[PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
            versions=VersionNumber[],dev_sources=String[],features)])
        lastprogress=Ref(0)
        progress(p)=if p["completed"]-lastprogress[]>=27
            lastprogress[]=p["completed"]
            println("K1 ",lastprogress[],"/",length(cases)," passed=",p["passed"]," failed=",p["failed"]," elapsed=",round(time()-started;digits=1));flush(stdout)
        end
        result=run_suite(suite;profile=:quick,strict=false,executor,progress_callback=progress)
        write_suite_json(result,paths["qualification"])
        open(paths["summary"],"a") do io
            for (case,run) in zip(cases,result.runs)
                run.status==:pass && continue
                reason=replace(run.message,','=>';','\n'=>' ')
                println(io,join((case.n,case.family,case.signature,case.query[1],case.horizon,case.method,case.passage,run.status,fill("",14)...,reason,
                    compact_case_id(case),compact_execution_id(case),compact_shard_id(case.n),compact_global_index(case)),','))
            end
        end
        protocol["completed"]=true;protocol["elapsed_seconds"]=time()-started;protocol["rss_bytes"]=Sys.maxrss()
        protocol["execution_state"]=suite_passed(result) ? "completed" : "completed_with_failures"
        protocol["source_unchanged"]=fingerprint()==initial;protocol["verdict"]=string(suite_verdict(result))
        protocol["started_execution_ids"]=started_ids;protocol["measured_execution_ids"]=measured_ids
        protocol["passed_execution_ids"]=passed_ids
        protocol["not_started_execution_ids"]=setdiff(compact_execution_id.(cases),started_ids)
        protocol["failed_execution_ids"]=[compact_execution_id(c) for (c,r) in zip(cases,result.runs) if r.status!=:pass]
        protocol["measured_global_indices"]=[compact_global_index(c) for c in cases if compact_execution_id(c) in measured_ids]
        open(io->TOML.print(io,protocol),paths["protocol"],"w")
        println("K1 verdict=",suite_verdict(result)," cases=",length(cases)," seconds=",round(time()-started;digits=2))
        protocol["source_unchanged"] || error("source_changed_during_run")
        suite_passed(result) || error("qualification_failed_see_summary")
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (2,3) || error("usage: binary_rank_compact_compare.jl --smoke|--screen|--decision_smoke|--decision_screen|--prepare|--full OUTPUT_DIRECTORY, or --shard DIMENSION OUTPUT_DIRECTORY")
    mode=Symbol(replace(ARGS[1],"--"=>""));mode in (:smoke,:screen,:decision_smoke,:decision_screen,:full,:prepare,:shard) || error("unknown mode")
    BLAS.set_num_threads(1)
    if mode==:shard
        length(ARGS)==3 || error("--shard requires dimension and output directory")
        compact_campaign(abspath(ARGS[3]);mode,shard_dimension=parse(Int,ARGS[2]))
    else
        length(ARGS)==2 || error("unexpected extra argument")
        output=abspath(ARGS[2])
        if mode==:full
            # The 900 s budget resets for each bounded, independently archived shard.
            # Stop on the first failed shard; already completed archives are kept.
            compact_campaign(output;mode=:prepare)
            for n in 2:128
                compact_campaign(joinpath(output,compact_shard_id(n));mode=:shard,shard_dimension=n)
                GC.gc()
            end
        else
            compact_campaign(output;mode)
        end
    end
end
