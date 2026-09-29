using PerfChecker, BenchmarkTools, Statistics, SHA, TOML
push!(LOAD_PATH,dirname(@__DIR__))
include("binary_rank_cases.jl")

function rank_campaign(output;mode=:smoke)
    Threads.nthreads()==1 || error("N01 initial protocol requires one Julia thread")
    mkpath(output)
    paths=Dict(key=>joinpath(output,"binary-rank-"*key*ext) for (key,ext) in
        (("samples",".csv"),("summary",".csv"),("protocol",".toml"),("qualification",".json")))
    any(isfile,values(paths)) && error("use a fresh output directory")
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"binary_rank.jl"),joinpath(@__DIR__,"binary_rank_cases.jl")],
        [joinpath(dirname(@__DIR__),"src",f) for f in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(f,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    initial=fingerprint();started=time();fixturecache=Dict{Tuple,Any}();artifacts=Dict{Tuple,Any}()
    dims=mode==:smoke ? (8,) : mode in (:screen,:episode_screen) ? RANK_SCREEN_DIMS : 2:128
    families=mode==:smoke ? (:low_rank_high_grade,) : RANK_FAMILIES
    records=[(;n,family,signature=RANK_SIGNATURES[mod1(index,3)]) for (index,n) in enumerate(dims) for family in families]
    passages=mode==:episode_screen ? (1:2) : (1:1)
    cases=[(;record...,horizon,method,query,passage) for passage in passages for record in records for horizon in RANK_HORIZONS
        for query in (mode in (:full,:prepare) ? ((:all,0),(:grade,1),(:coefficient,0)) : ((:all,0),))
        for method in (isodd(passage) ? RANK_METHODS : reverse(RANK_METHODS))]
    protocol=Dict("technique"=>"N01","mode"=>string(mode),"timing_status"=>"exploratory_Etendue3D_external_contention_uncontrolled",
        "threads"=>1,"blas_threads"=>1,"julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,
        "source_sha256"=>initial,"horizons"=>collect(RANK_HORIZONS),"screen_dimensions"=>collect(RANK_SCREEN_DIMS),
        "integer_dimension_sequences"=>Dict(string(f)=>collect(2:128) for f in RANK_FAMILIES),
        "families"=>collect(string.(RANK_FAMILIES)),"signatures"=>collect(string.(RANK_SIGNATURES)),
        "coefficient_period"=>4,"coefficient_values"=>collect(RANK_COEFFICIENTS),
        "output_contract"=>"H owned ambient sparse multivectors retained until observation returns; same four cycling input values; all calls include validation and encoding/decoding",
        "build_contract"=>"new rank/prepared structure per build sample; compilation first call reported separately; fixture and Int64 oracle excluded",
        "episode_contract"=>"each observed call builds a new artifact then executes H products and retains all H owned outputs; fixture and oracle excluded; warm compiled code",
        "additive_model"=>"(median_build + median_hot_batch) / H; descriptive estimate only, never a measured end-to-end cost",
        "passages"=>collect(passages),"method_order_by_passage"=>Dict(string(p)=>collect(string.(isodd(p) ? RANK_METHODS : reverse(RANK_METHODS))) for p in passages),
        "max_rank"=>16,"max_pairs"=>65_536,"max_plan_estimated_bytes"=>64<<20,
        "rss_limit_bytes"=>2<<30,"wall_limit_seconds"=>900,"samples"=>7,
        "method_order"=>collect(string.(RANK_METHODS)),"cases"=>[Dict("dimension"=>c.n,"family"=>string(c.family),"signature"=>string(c.signature),
            "horizon"=>c.horizon,"method"=>string(c.method),"query"=>string(c.query),"passage"=>c.passage) for c in cases])
    open(io->TOML.print(io,protocol),paths["protocol"],"w")
    mode==:prepare && return nothing
    open(paths["samples"],"w") do io
        println(io,"dimension,family,signature,query,horizon,method,passage,phase,sample,time_ns,gc_ns,memory_bytes,allocations")
    end
    open(paths["summary"],"w") do io
        println(io,"dimension,family,signature,query,horizon,method,passage,status,rank,support_size,plan_bytes,first_build_us,build_median_us,hot_median_us,additive_model_us_per_product,episode_median_us,episode_measured_us_per_product,reason")
    end
    function append_samples(case,phase,trial)
        open(paths["samples"],"a") do io
            for i in eachindex(trial.times)
                println(io,join((case.n,case.family,case.signature,case.query[1],case.horizon,case.method,case.passage,phase,i,trial.times[i],trial.gctimes[i],trial.memory,trial.allocs),','))
            end
        end
    end
    function executor(planned,config,setup,workload)
        time()-started<=900 || error("campaign_wall_budget")
        Sys.maxrss()<=2<<30 || error("campaign_rss_budget")
        case=planned.feature.options[:rank_case]
        key=(case.n,case.family,case.signature)
        fixture=get!(fixturecache,key) do;rank_fixture(key...);end
        artifactkey=(key...,case.method,case.query,case.passage)
        cached=get!(artifacts,artifactkey) do
            firstbuild=@timed rank_artifact(fixture,case.method,case.query)
            artifact=firstbuild.value
            rank_correct(fixture,case.method,artifact,case.query) || error("independent_Int64_oracle_failed")
            build=@benchmark rank_artifact($fixture,$(case.method),$(case.query)) samples=7 evals=1 seconds=0.02
            append_samples(case,:build,build)
            (;artifact,firstbuild_us=1e6*firstbuild.time,build_us=median(build.times)/1000)
        end
        artifact=cached.artifact;method=case.method;query=case.query;horizon=case.horizon
        warm=rank_owned_execute(fixture,method,artifact,query,horizon)
        rank_owned_correct(fixture,warm,query,horizon) || error("hot_owned_episode_oracle_failed")
        trial=@benchmark rank_owned_execute($fixture,$method,$artifact,$query,$horizon) samples=7 evals=1 seconds=0.02
        append_samples(case,:hot,trial)
        episodewarm=rank_owned_episode(fixture,method,query,horizon)
        rank_owned_correct(fixture,episodewarm,query,horizon) || error("complete_episode_oracle_failed")
        episode=@benchmark rank_owned_episode($fixture,$method,$query,$horizon) samples=7 evals=1 seconds=0.02
        append_samples(case,:episode,episode)
        rank_owned_correct(fixture,rank_owned_episode(fixture,method,query,horizon),query,horizon) || error("post_timing_episode_oracle_failed")
        rank_correct(fixture,method,artifact,query) || error("post_timing_oracle_failed")
        open(paths["summary"],"a") do io
            println(io,join((case.n,case.family,case.signature,query[1],horizon,method,case.passage,:pass,fixture.d,length(fixture.masks),
                Base.summarysize(artifact),cached.firstbuild_us,cached.build_us,median(trial.times)/1000,
                (cached.build_us+median(trial.times)/1000)/horizon,median(episode.times)/1000,median(episode.times)/1000/horizon,""),','))
        end
        # The primary PerfChecker table now describes complete episodes.
        # Hot and construction observations remain in the phase-tagged CSV.
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,[:N01,:exploratory,:complete_episode],
            [PerfChecker.PackageSpec(name="Garamon")],[Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"independent Int64 ordered-word oracle; every owned episode output and checksum checked; four coefficient phases"),"source_fingerprint"=>initial,
                "execution"=>Dict("mode"=>"shared_process_exploratory","threads"=>1))])
    end
    mktempdir(;prefix="rank-perfchecker-") do temporary
        entry=joinpath(temporary,"rank.jl");write(entry,"# N01 shared-process executor entrypoint\n")
        features=[FeatureSpec(Symbol(:rank_,i);entrypoint=entry,backend=:benchmark,oracle=OracleSpec(),
            comparison_key="N01/owned_episode/$(c.n)/$(c.family)/$(c.signature)/$(c.query)/$(c.horizon)/pass$(c.passage)",options=Dict(:rank_case=>c)) for (i,c) in enumerate(cases)]
        suite=SoftwareSuite(:binary_rank,[PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
            versions=VersionNumber[],dev_sources=String[],features)])
        lastprogress=Ref(0)
        progress(p)=if p["completed"]-lastprogress[]>=27
            lastprogress[]=p["completed"];println("N01 ",lastprogress[],"/",length(cases)," elapsed=",round(time()-started;digits=1));flush(stdout)
        end
        result=run_suite(suite;profile=:quick,strict=false,executor,progress_callback=progress)
        write_suite_json(result,paths["qualification"])
        open(paths["summary"],"a") do io
            for (case,run) in zip(cases,result.runs)
                run.status==:pass && continue
                reason=replace(run.message,','=>';','\n'=>' ')
                println(io,join((case.n,case.family,case.signature,case.query[1],case.horizon,case.method,case.passage,run.status,fill("",9)...,reason),','))
            end
        end
        protocol["completed"]=true;protocol["elapsed_seconds"]=time()-started;protocol["rss_bytes"]=Sys.maxrss()
        protocol["source_unchanged"]=fingerprint()==initial;protocol["verdict"]=string(suite_verdict(result))
        open(io->TOML.print(io,protocol),paths["protocol"],"w")
        println("N01 verdict=",suite_verdict(result)," cases=",length(cases)," seconds=",round(time()-started;digits=2))
        protocol["source_unchanged"] || error("source_changed_during_run")
        suite_passed(result) || error("qualification_failed_see_summary")
    end
end

length(ARGS)==2 || error("usage: binary_rank_compare.jl --smoke|--screen|--episode_screen|--full|--prepare OUTPUT_DIRECTORY")
mode=Symbol(replace(ARGS[1],"--"=>""));mode in (:smoke,:screen,:episode_screen,:full,:prepare) || error("unknown mode")
BLAS.set_num_threads(1)
rank_campaign(abspath(ARGS[2]);mode)
