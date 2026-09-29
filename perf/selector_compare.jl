using PerfChecker, Statistics, SHA, TOML
include("selector_cases.jl")

function se_hash_tree(directory)
    files=sort(filter(path->endswith(path,".jl"),readdir(directory;join=true)))
    bytes2hex(sha256(join((basename(path)*"\n"*read(path,String) for path in files),"\n")))
end

function se_campaign(output;smoke=false,reverse_order=false,prepare_only=false)
    mkpath(output)
    manifest_path=joinpath(output,"selector-instances.toml")
    samples_path=joinpath(output,"selector-samples.csv")
    isfile(samples_path) && error("choose a directory without previous selector samples")
    records=se_manifest()
    counts=Dict(string(phase)=>length(unique(r.group for r in records if r.phase==phase)) for phase in (:train,:validation,:test))
    groups=Dict(phase=>Set(r.group for r in records if r.phase==phase) for phase in (:train,:validation,:test))
    @assert isempty(intersect(groups[:train],union(groups[:validation],groups[:test])))
    @assert isempty(intersect(groups[:validation],groups[:test]))
    split_evaluable=counts["train"]>=20 && counts["validation"]>=2 && counts["test"]>=2
    manifest=Dict("model"=>"analytic_v1_unfitted","learned_model"=>false,
        "split_evaluable"=>split_evaluable,"independent_groups"=>counts,
        "test_algebra_family"=>"many_null","test_support_family"=>"random_sparse",
        "validation_algebra_family"=>"scaled_diagonal","validation_support_family"=>"subalgebra",
        "coefficient_period"=>4,"coefficient_values"=>[-2,-1,1,2],
        "coefficient_rule"=>"values[1+mod(iteration+operand+slot,4)]",
        "instances"=>[Dict("id"=>r.id,"algebra_family"=>string(r.algebra_family),
            "support_family"=>string(r.support_family),"template"=>r.template,"dimension"=>r.n,
            "input_masks_decimal"=>[string.(s) for s in r.supports],"metric_diagonal"=>se_diagonal(r),
            "ancestry"=>r.ancestry,"invariant"=>r.invariant,"group"=>r.group,"split"=>string(r.phase)) for r in records])
    open(io->TOML.print(io,manifest),manifest_path,"w")
    open(joinpath(output,"selector-replay-matrix.csv"),"w") do io
        println(io,"case,instance,group,split,horizon,outputs,strategy,threads,cache_state")
        for record in records,horizon in (1,32,1024),q in (1,4),strategy in (:full,:recursive,:join3,:prepared,:workspace,:auto)
            id="$(record.id)-h$horizon-q$q-$strategy"
            println(io,join((id,record.id,record.group,record.phase,horizon,q,strategy,1,"warm_code_new_structure"),','))
        end
    end
    if prepare_only
        println("Prepared 144 structural instances and 5184 replay cases; independent groups ",counts,
            "; learned-model split evaluable = ",split_evaluable)
        return nothing
    end
    # Freeze a small, predeclared stratified screening slice before any timing.
    selected=NamedTuple[]
    for phase in (:train,:validation,:test)
        seen=Set{String}()
        for record in records
            record.phase==phase && !(record.group in seen) || continue
            push!(selected,record);push!(seen,record.group)
            length(seen)>=2 && break
        end
    end
    smoke && (selected=selected[1:1])
    horizons=smoke ? (1,) : (1,32)
    requests=smoke ? (1,) : (1,4)
    cases=[(record,horizon,q,strategy) for record in selected for horizon in horizons
        for q in requests for strategy in (:full,:recursive,:join3,:prepared,:workspace,:auto)]
    reverse_order && reverse!(cases)
    length(cases)<=300 || error("selector slice exceeds case budget")
    root=dirname(@__DIR__); source_hash=se_hash_tree(joinpath(root,"src"))
    setup_results=Dict{String,Any}[]
    mktempdir(;prefix="garamon-selector-") do temporary
        features=FeatureSpec[]; traces=String[]
        for (i,(record,horizon,q,strategy)) in enumerate(cases)
            entry=joinpath(temporary,"case-$i.jl");trace=joinpath(temporary,"trace-$i.toml")
            push!(traces,trace)
            open(entry,"w") do io
                println(io,"include(",repr(joinpath(@__DIR__,"selector_cases.jl")),")")
                println(io,"perf_setup()=se_setup(",repr(record.id),",",horizon,",",q,",:",strategy,",",repr(trace),")")
                println(io,"perf_workload(state)=se_episode(state.fixture,state.strategy)")
                println(io,"perf_oracle(state)=perf_workload(state)==state.fixture.expected")
            end
            push!(features,FeatureSpec(Symbol(:selector_,i);backend=:benchmark,entrypoint=entry,
                description="exact owned triple episode; $(record.phase); $strategy",
                comparison_key="selector/owned/warm_new/$(record.id)/$horizon/$q",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>0.1,:threads=>1)))
        end
        package=PackageSuite("Garamon";source=root,worker_environment=joinpath(@__DIR__,"runner"),
            versions=VersionNumber[],dev_sources=String[],features)
        result=run_suite(SoftwareSuite(:selector_exact,[package]);profile=:quick,strict=false)
        write_suite_json(result,joinpath(output,"selector-qualification.json"))
        open(samples_path,"w") do io
            println(io,"instance,group,split,horizon,outputs,strategy,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations")
            for (run,(record,horizon,q,strategy)) in zip(result.runs,cases)
                if run.status==:pass
                    table=only(run.result.tables)
                    for j in eachindex(table.times)
                        println(io,join((record.id,record.group,record.phase,horizon,q,strategy,run.status,j,
                            table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((record.id,record.group,record.phase,horizon,q,strategy,run.status,"","","","",""),','))
                end
            end
        end
        for trace in traces
            isfile(trace) && push!(setup_results,TOML.parsefile(trace))
        end
        open(io->TOML.print(io,Dict("observations"=>setup_results)),joinpath(output,"selector-diagnostics.toml"),"w")
        suite_passed(result) || error("selector screening contains a failed or refused candidate; see qualification")
        println("Selector: ",length(result.runs)," exact cases; independent split groups ",counts,
            "; learned-model split evaluable = ",split_evaluable);flush(stdout)
    end
    source_hash==se_hash_tree(joinpath(root,"src")) || error("source changed during selector measurement")
    environment=Dict("julia"=>string(VERSION),"perfchecker"=>string(pkgversion(PerfChecker)),
        "cpu"=>Sys.cpu_info()[1].model,"threads"=>Threads.nthreads(),"model"=>"analytic_v1_unfitted",
        "kernel_sha256"=>source_hash,"instances_sha256"=>bytes2hex(sha256(read(manifest_path))),
        "reverse_order"=>reverse_order,"cold_comparison"=>false,"preexisting_artifact_comparison"=>false,
        "harness_sha256"=>bytes2hex(sha256(vcat(read(@__FILE__),read(joinpath(@__DIR__,"selector_cases.jl"))))),
        "selector_sha256"=>bytes2hex(sha256(read(joinpath(root,"src","selector.jl")))),
        "manifests"=>Dict(path=>isfile(path) ? bytes2hex(sha256(read(path))) : "absent"
            for path in (joinpath(root,"Manifest.toml"),joinpath(@__DIR__,"controller","Manifest.toml"),joinpath(@__DIR__,"runner","Manifest.toml"))))
    open(io->TOML.print(io,environment),joinpath(output,"selector-environment.toml"),"w")
end

if abspath(PROGRAM_FILE)==@__FILE__
    smoke="--smoke" in ARGS; reverse_order="--reverse" in ARGS;prepare_only="--prepare-only" in ARGS
    paths=filter(arg->!(arg in ("--smoke","--reverse","--prepare-only")),ARGS)
    length(paths)==1 || error("usage: selector_compare.jl [--smoke] [--reverse] [--prepare-only] OUTPUT_DIRECTORY")
    se_campaign(abspath(only(paths));smoke,reverse_order,prepare_only)
end
