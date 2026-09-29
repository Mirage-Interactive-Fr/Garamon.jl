using SHA,TOML
if abspath(PROGRAM_FILE)==@__FILE__
    using PerfChecker
end
include("n05_compare.jl")
include("n05_shared_workspace.jl")

function n05_shared_hash()
    files=sort(vcat(filter(f->endswith(f,".jl"),readdir(joinpath(dirname(@__DIR__),"src");join=true)),
        [joinpath(@__DIR__,name) for name in ("n05_pfaffian.jl","n05_compare.jl","n05_shards.jl","n05_shared_workspace.jl","n05_shared_compare.jl")]))
    bytes2hex(sha256(join((basename(f)*"\n"*read(f,String) for f in files),"\n")))
end

function n05_shared_setup(n,k,count,horizon,family,sharing,changes,strategy,trace)
    BLAS.set_num_threads(1)
    state=n05_shared_fixture(n,k,count,horizon,family,sharing,changes)
    first=@timed Base.invokelatest(n05_shared_episode,state,strategy)
    n05_shared_qualify(state,first.value) || error("K3 numerical oracle failure")
    # Counters/retained-size diagnostics use a separate, untimed replay.
    diagnostic=n05_shared_episode(state,strategy;trace=true)
    Sys.maxrss()<=2<<30 || error("K3 worker RSS checkpoint budget")
    if !isfile(trace)
        r=diagnostic
        data=Dict("n"=>n,"chain_length"=>k,"chains"=>count,"horizon"=>horizon,"metric"=>string(family),
            "sharing"=>string(sharing),"changes"=>string(changes),"strategy"=>string(strategy),
            "first_episode_ms"=>1000first.time,"compile_ms"=>1000first.compile_time,"first_bytes"=>first.bytes,
            "contraction_builds"=>r.builds,"pair_slots"=>r.pair_count,"retained_bytes"=>r.retained_bytes,
            "worker_peak_rss_bytes"=>Sys.maxrss(),"qualified"=>true,"counters_from_separate_untimed_replay"=>true,
            "max_absolute_error"=>string(maximum(abs(BigFloat(first.value[j,t])-BigFloat(state.exact[1+mod(t-1,3)][j]))
                for t in 1:state.horizon for j in eachindex(state.chains))),
            "exact_phase_values"=>[string.(values) for values in state.exact])
        open(io->TOML.print(io,data),trace,"w")
    end
    state
end

function n05_shared_campaign(output;prepare_only=false,reverse_order=false)
    condition=n05_condition();started=time();mkpath(output)
    isfile(joinpath(output,"n05-shared-samples.csv")) && error("choose a fresh K3 result directory")
    count=0
    open(joinpath(output,"n05-shared-replay-matrix.csv"),"w") do io
        println(io,"n,chain_length,chains,horizon,metric,sharing,changes,strategy,threads")
        for n in (2,4,8,32,65,129),k in (2,4,8,16),q in (1,4,16),h in (1,32,1024),
            family in (:euclidean,:signed,:null,:general),sharing in (:shared,:disjoint),
            changes in (:stable,:all,:one),strategy in N05_SHARED_ROUTES
            println(io,join((n,k,q,h,family,sharing,changes,strategy,1),','));count+=1
        end
    end
    prepare_only && return println("K3 prepared ",count," identities; no timing")
    fixtures=((4,4,4,32,:signed,:shared,:all),(65,8,4,32,:signed,:shared,:stable),
        (4,2,1,1,:euclidean,:disjoint,:stable),(4,4,4,3,:general,:disjoint,:one))
    cases=[(;n,k,q,h,family,sharing,changes,strategy) for (n,k,q,h,family,sharing,changes) in fixtures
        for strategy in N05_SHARED_ROUTES]
    reverse_order && reverse!(cases)
    fingerprint=n05_shared_hash();details=Dict{String,Any}[]
    mktempdir(;prefix="garamon-n05-shared-") do temporary
        features=FeatureSpec[];traces=String[]
        for (i,r) in enumerate(cases)
            entry=joinpath(temporary,"case-$i.jl");trace=joinpath(temporary,"trace-$i.toml");push!(traces,trace)
            open(entry,"w") do io
                println(io,"include(",repr(@__FILE__),")")
                println(io,"perf_setup()=n05_shared_setup(",r.n,",",r.k,",",r.q,",",r.h,",:",r.family,",:",r.sharing,",:",r.changes,",:",r.strategy,",",repr(trace),")")
                println(io,"perf_workload(state)=n05_shared_episode(state,:",r.strategy,")")
                println(io,"perf_oracle(state)=Sys.maxrss()<=2<<30 && n05_shared_qualify(state,perf_workload(state))")
            end
            push!(features,FeatureSpec(Symbol(:n05_shared_,i);backend=:benchmark,entrypoint=entry,
                description="K3 shared scalar chains; owned output; preparation included; $(r.strategy)",
                comparison_key="K3/owned/$(r.n)/$(r.k)/$(r.q)/$(r.h)/$(r.family)/$(r.sharing)/$(r.changes)",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>0.1,:threads=>1)))
        end
        package=PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
            versions=VersionNumber[],dev_sources=String[],features)
        result=run_suite(SoftwareSuite(:n05_shared,[package]);profile=:quick,strict=false)
        write_suite_json(result,joinpath(output,"n05-shared-qualification.json"))
        open(joinpath(output,"n05-shared-samples.csv"),"w") do io
            println(io,"n,chain_length,chains,horizon,metric,sharing,changes,strategy,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations")
            for (run,r) in zip(result.runs,cases)
                prefix=(r.n,r.k,r.q,r.h,r.family,r.sharing,r.changes,r.strategy,run.status)
                if run.status==:pass
                    table=only(run.result.tables)
                    for j in eachindex(table.times)
                        println(io,join((prefix...,j,table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((prefix...,"","","","",""),','))
                end
            end
        end
        for trace in traces;isfile(trace)&&push!(details,TOML.parsefile(trace));end
        open(io->TOML.print(io,Dict("observations"=>details)),joinpath(output,"n05-shared-diagnostics.toml"),"w")
        suite_passed(result) || error("K3 qualification failed; partial results preserved")
    end
    fingerprint==n05_shared_hash() || error("K3 sources changed during campaign")
    time()-started<=900 || error("K3 wall-time checkpoint budget")
    environment=Dict("measurement_condition"=>condition.condition,"interference_label"=>condition.label,
        "condition_declaration"=>condition.source,"concurrent_external_jobs"=>condition.concurrent_external_jobs,
        "source_sha256"=>fingerprint,"elapsed_seconds"=>time()-started,"threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(),"julia"=>string(VERSION),"perfchecker"=>string(pkgversion(PerfChecker)),
        "cpu"=>Sys.cpu_info()[1].model,"reverse_order"=>reverse_order,"cases"=>length(cases),
        "controller_manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"controller","Manifest.toml")))))
    open(io->TOML.print(io,environment),joinpath(output,"n05-shared-environment.toml"),"w")
    sum(filesize,filter(f->startswith(basename(f),"n05-shared-")&&isfile(f),readdir(output;join=true));init=0)<=64<<20 || error("K3 result disk budget")
    println("K3: ",length(cases)," qualified routes")
end

if abspath(PROGRAM_FILE)==@__FILE__
    flags=filter(arg->startswith(arg,"--"),ARGS)
    all(flag->flag in ("--prepare-only","--smoke","--reverse"),flags) || error("unknown K3 flag")
    ("--prepare-only" in flags || "--smoke" in flags) || error("choose --prepare-only or --smoke explicitly")
    paths=filter(arg->!startswith(arg,"--"),ARGS);length(paths)==1 || error("one K3 output directory required")
    n05_shared_campaign(abspath(only(paths));prepare_only="--prepare-only" in flags,reverse_order="--reverse" in flags)
end
