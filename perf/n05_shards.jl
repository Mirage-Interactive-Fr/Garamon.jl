# Stable execution of the original 36,480-case N05 grid. No grid reordering.
function n05_grid()
    [(;case_id=i,r...) for (i,r) in enumerate([(;n,k,horizon,family,strategy)
        for n in 2:129 for k in (2,4,6,8,16) for horizon in (1,32,1024)
        for family in (:euclidean,:signed,:null,:zero,:general)
        for strategy in (:full,:recursive,:pfaffian,:precontracted)
        if strategy!=:recursive || family!=:general])]
end

function n05_grid_fingerprint(grid=n05_grid())
    canonical="N05-grid-v1;numeric=Float64;oracle=Rational{BigInt}\n"*
        join((join((r.case_id,r.n,r.k,r.horizon,r.family,r.strategy),',') for r in grid),"\n")*"\n"
    bytes2hex(sha256(canonical))
end

function n05_select_cases(grid;shard=nothing,interval=nothing,max_cases=32)
    (shard===nothing) ⊻ (interval===nothing) || throw(ArgumentError("choose exactly one shard or interval"))
    selected=if shard!==nothing
        index,count=shard
        1<=index<=count<=length(grid) || throw(ArgumentError("shard must satisfy 1 <= index <= count <= grid size"))
        filter(r->mod(r.case_id-1,count)==index-1,grid)
    else
        first,last=interval
        1<=first<=last<=length(grid) || throw(ArgumentError("interval outside N05 grid"))
        grid[first:last]
    end
    length(selected)<=max_cases || throw(ArgumentError("N05 shard exceeds $max_cases cases; increase shard count or shorten interval"))
    selected
end

function n05_parse_selection(flags)
    shard=nothing;interval=nothing
    for flag in flags
        if startswith(flag,"--shard=")
            shard===nothing || throw(ArgumentError("duplicate shard"))
            parts=split(chopprefix(flag,"--shard="),'/')
            length(parts)==2 || throw(ArgumentError("use --shard=INDEX/COUNT"))
            shard=Tuple(parse.(Int,parts))
        elseif startswith(flag,"--range=")
            interval===nothing || throw(ArgumentError("duplicate interval"))
            parts=split(chopprefix(flag,"--range="),':')
            length(parts)==2 || throw(ArgumentError("use --range=FIRST:LAST"))
            interval=Tuple(parse.(Int,parts))
        elseif !(flag in ("--smoke","--prepare-only","--reverse"))
            throw(ArgumentError("unknown N05 option: $flag"))
        end
    end
    shard!==nothing && interval!==nothing && throw(ArgumentError("shard and range are mutually exclusive"))
    "--smoke" in flags && (shard!==nothing || interval!==nothing) && throw(ArgumentError("smoke and manifest selection are mutually exclusive"))
    (;shard,interval)
end

"""Audit one repetition root; do not mix forward/reverse repetitions in it."""
function n05_audit_shards(root)
    files=[joinpath(path,"n05-shard-environment.toml") for path in readdir(root;join=true)
        if isdir(path)&&isfile(joinpath(path,"n05-shard-environment.toml"))]
    records=TOML.parsefile.(files);grid=n05_grid();fingerprint=n05_grid_fingerprint(grid)
    frequencies=Dict{Int,Int}();failed=Int[];missing_files=String[]
    for (record,file) in zip(records,files)
        record["grid_sha256"]==fingerprint || error("mixed or changed N05 grid")
        append!(failed,record["failed_case_ids"])
        for id in record["completed_case_ids"]
            1<=id<=length(grid) || error("N05 case id outside grid")
            frequencies[id]=get(frequencies,id,0)+1
            for path in (joinpath(dirname(file),"n05-case-$id-qualification.json"),joinpath(dirname(file),"n05-shard-samples.csv"))
                isfile(path)||push!(missing_files,path)
            end
        end
    end
    missing=setdiff(collect(1:length(grid)),collect(keys(frequencies)))
    duplicates=sort([id for (id,count) in frequencies if count>1])
    contexts=unique((r["source_sha256"],r["measurement_condition"],r["interference_label"])
        for r in records if !isempty(r["completed_case_ids"]))
    complete=isempty(missing)&&isempty(duplicates)&&isempty(failed)&&isempty(missing_files)&&length(contexts)==1
    (;complete,covered=length(frequencies),missing,duplicates,failed=sort(unique(failed)),missing_files,contexts)
end

function n05_shard_campaign(output;shard=nothing,interval=nothing,prepare_only=false,reverse_order=false)
    condition=n05_condition();grid=n05_grid();cases=n05_select_cases(grid;shard,interval)
    name=shard===nothing ? "n05-range-$(lpad(interval[1],5,'0'))-$(lpad(interval[2],5,'0'))" :
        "n05-shard-$(lpad(shard[1],4,'0'))-of-$(lpad(shard[2],4,'0'))"
    destination=joinpath(output,name);mkpath(destination)
    samples=joinpath(destination,"n05-shard-samples.csv")
    isfile(samples) && error("N05 shard already has samples; choose another result root")
    reverse_order && reverse!(cases)
    open(joinpath(destination,"n05-shard-selection.csv"),"w") do io
        println(io,"case_id,n,chain_length,horizon,metric,strategy,numeric_type,construction_included")
        for r in cases
            println(io,join((r.case_id,r.n,r.k,r.horizon,r.family,r.strategy,"Float64",r.strategy!=:precontracted),','))
        end
    end
    started=time();fingerprint=n05_source_hash();metadata=Dict{String,Any}(
        "grid_schema"=>"N05-grid-v1","grid_sha256"=>n05_grid_fingerprint(grid),"grid_cases"=>length(grid),
        "selected_case_ids"=>getproperty.(cases,:case_id),"source_sha256"=>fingerprint,
        "measurement_condition"=>condition.condition,"condition_declaration"=>condition.source,
        "interference_label"=>condition.label,"concurrent_external_jobs"=>condition.concurrent_external_jobs,
        "selection"=>name,"reverse_order"=>reverse_order,"completed"=>false,"status"=>"prepared",
        "completed_case_ids"=>Int[],"current_case_id"=>0,"failed_case_ids"=>Int[],
        "numeric_type"=>"Float64","oracle"=>"Rational{BigInt} exterior action",
        "float_atol"=>1e-8,"float_rtol"=>1e-10,"max_cases"=>32,"wall_seconds"=>900,
        "worker_rss_checkpoint_bytes"=>2<<30,"disk_bytes"=>64<<20,"threads"=>1)
    meta_path=joinpath(destination,"n05-shard-environment.toml")
    save_metadata()=open(io->TOML.print(io,metadata),meta_path,"w")
    save_metadata()
    prepare_only && return println("Prepared ",length(cases)," cases in ",destination)
    Threads.nthreads()==1 || error("N05 shards require JULIA_NUM_THREADS=1")
    metadata["julia"]=string(VERSION);metadata["perfchecker"]=string(pkgversion(PerfChecker))
    metadata["controller_manifest_sha256"]=bytes2hex(sha256(read(joinpath(@__DIR__,"controller","Manifest.toml"))))
    metadata["cpu"]=Sys.cpu_info()[1].model
    open(samples,"w") do io
        println(io,"case_id,n,chain_length,horizon,metric,strategy,construction_included,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations")
    end
    mktempdir(;prefix="garamon-n05-shard-") do temporary
        for r in cases
            if time()-started>=840
                metadata["status"]="wall_budget_before_next_case";save_metadata();break
            end
            metadata["current_case_id"]=r.case_id;metadata["status"]="running";save_metadata()
            entry=joinpath(temporary,"case.jl");trace=joinpath(destination,"n05-case-$(r.case_id)-diagnostics.toml")
            open(entry,"w") do io
                println(io,"include(",repr(joinpath(@__DIR__,"n05_compare.jl")),")")
                println(io,"perf_setup()=n05_setup(",r.n,",",r.k,",",r.horizon,",:",r.family,",:",r.strategy,",",repr(trace),")")
                println(io,"perf_workload(state)=n05_episode(state,:",r.strategy,")")
                println(io,"perf_oracle(state)=n05_qualify(state,perf_workload(state))")
            end
            contract=r.strategy==:precontracted ? "preexisting_contractions" : "coordinates_construction_included"
            feature=FeatureSpec(Symbol(:n05_case_,r.case_id);backend=:benchmark,entrypoint=entry,
                description="N05 grid v1 case $(r.case_id), $contract",
                comparison_key="n05/$contract/$(r.n)/$(r.k)/$(r.horizon)/$(r.family)",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>0.1,:threads=>1))
            package=PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
                versions=VersionNumber[],dev_sources=String[],features=[feature])
            result=run_suite(SoftwareSuite(:n05_shard,[package]);profile=:quick,strict=false)
            write_suite_json(result,joinpath(destination,"n05-case-$(r.case_id)-qualification.json"))
            run=only(result.runs)
            open(samples,"a") do io
                prefix=(r.case_id,r.n,r.k,r.horizon,r.family,r.strategy,r.strategy!=:precontracted,run.status)
                if run.status==:pass
                    table=only(run.result.tables)
                    for j in eachindex(table.times)
                        println(io,join((prefix...,j,table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((prefix...,"","","","",""),','))
                    push!(metadata["failed_case_ids"],r.case_id)
                end
            end
            push!(metadata["completed_case_ids"],r.case_id);metadata["current_case_id"]=0
            metadata["elapsed_seconds"]=time()-started;save_metadata()
            sum(filesize,filter(isfile,readdir(destination;join=true));init=0)<=64<<20 || error("N05 shard disk budget")
            fingerprint==n05_source_hash() || error("N05 shard sources changed")
        end
    end
    metadata["completed"]=length(metadata["completed_case_ids"])==length(cases)
    metadata["completed"] && (metadata["status"]=isempty(metadata["failed_case_ids"]) ? "qualified" : "qualification_failures")
    metadata["elapsed_seconds"]=time()-started;save_metadata()
    metadata["completed"] && isempty(metadata["failed_case_ids"]) || error("N05 shard incomplete or unqualified; partial results retained")
    println("N05 shard qualified: ",length(cases)," cases; ",destination)
end
