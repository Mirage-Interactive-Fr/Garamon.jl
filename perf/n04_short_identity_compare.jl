# Bounded N04 controller. --prepare-only never invokes PerfChecker or @timed.
using SHA,TOML
if abspath(PROGRAM_FILE)==(@__FILE__) && !("--prepare-only" in ARGS)
    using PerfChecker
end
include("n04_short_identity.jl")

function n04_condition(environment=ENV)
    haskey(environment,"GARAMONBENCH_CONDITION") || throw(ArgumentError("declare GARAMONBENCH_CONDITION explicitly"))
    condition=strip(environment["GARAMONBENCH_CONDITION"])
    condition in ("exploratory_interference","isolated") || throw(ArgumentError("unknown measurement condition"))
    label=strip(get(environment,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition=="exploratory_interference" && isempty(label) && throw(ArgumentError("exploration requires an interference label"))
    condition=="isolated" && !isempty(label) && throw(ArgumentError("isolated condition requires an empty label"))
    length(label)<=256 && !any(iscntrl,label) || throw(ArgumentError("invalid interference label"))
    (;condition,label,declaration="environment",automatically_verified_isolation=false)
end

function n04_runtime_check(;version=VERSION,threads=Threads.nthreads())
    version.major==1 && version.minor==13 || throw(ArgumentError("official N04 runs require Julia 1.13"))
    threads==1 || throw(ArgumentError("N04 controller requires exactly one Julia thread"))
    true
end

function n04_source_files(root=dirname(@__DIR__))
    files=vcat([joinpath("src",f) for f in readdir(joinpath(root,"src")) if endswith(f,".jl")],
        ["Project.toml","Manifest.toml","perf/controller/Project.toml","perf/controller/Manifest.toml",
         "perf/runner/Project.toml","perf/n04_short_identity.jl","perf/n04_short_identity_compare.jl",
         "perf/n04_short_identity_protocol.toml"])
    isfile(joinpath(root,"perf/runner/Manifest.toml")) && push!(files,"perf/runner/Manifest.toml")
    all(f->isfile(joinpath(root,f)),files) || error("missing N04 source/environment file")
    sort!(files)
end

n04_files_hash(root,files)=bytes2hex(sha256(join((f*"\n"*read(joinpath(root,f),String) for f in sort(files)),"\n")))
n04_source_hash()=n04_files_hash(dirname(@__DIR__),n04_source_files())
n04_grid_hash(records=n04_grid())=bytes2hex(sha256("N04-grid-v1\n"*join((join(values(r),',') for r in records),"\n")))

function n04_select(records=n04_grid();smoke=false,case_ids=nothing,reverse_order=false)
    smoke && case_ids!==nothing && throw(ArgumentError("choose smoke or explicit case IDs"))
    selected=if smoke
        fixtures=((2,64,1,:euclidean,:vector,:stable),(4,256,32,:signed,:mixed_positive,:stable),
            (65,1024,32,:null,:vector,:coefficients),(4,16,1,:general,:negative_scalar_vector,:stable))
        reduce(vcat,[filter(r->(r.n,r.p,r.H,r.family,r.shape,r.changes)==f,records) for f in fixtures])
    elseif case_ids!==nothing
        1<=length(case_ids)<=20 && length(unique(case_ids))==length(case_ids) || throw(ArgumentError("select 1..20 distinct case IDs"))
        all(id->1<=id<=length(records),case_ids) || throw(ArgumentError("case ID outside N04 grid"))
        [records[id] for id in case_ids]
    else
        NamedTuple[]
    end
    reverse_order && reverse!(selected)
    selected
end

function n04_resource_check(state,output;rss=Sys.maxrss(),max_rss=2<<30,max_fixture=64<<20,max_output=64<<20)
    rss<=max_rss || error("N04 worker RSS checkpoint exceeded")
    Base.summarysize(state)<=max_fixture || error("N04 fixture bytes exceeded")
    Base.summarysize(output)<=max_output || error("N04 owned-output bytes exceeded")
    true
end

function n04_output_check(state,output)
    output isa AbstractVector && length(output)==state.H && all(x->x isa N04Terms,output) || return false
    length(unique(objectid.(output)))==length(output) || return false
    any(out===input for out in output for input in state.inputs) && return false
    n04_qualify(state,output)
end

function n04_worker_setup(record,trace,expected_hash)
    n04_runtime_check();BLAS.set_num_threads(1)
    n04_source_hash()==expected_hash || error("N04 worker source/environment drift")
    fixture=@timed n04_fixture(record.n,record.p,record.H,record.family,record.shape,record.changes)
    state=fixture.value
    first_episode=@timed Base.invokelatest(n04_episode,state,record.strategy)
    n04_resource_check(state,first_episode.value)
    n04_output_check(state,first_episode.value) || error("N04 exact output/ownership oracle failure")
    diagnostic=n04_episode(state,record.strategy;trace=true)
    n04_resource_check(state,diagnostic.output)
    details=Dict("case_id"=>record.case_id,"source_sha256"=>expected_hash,"julia"=>string(VERSION),
        "worker_pid"=>getpid(),"first_episode_scope"=>"first invocation after exact fixture setup in this worker; not process startup",
        "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"oracle_exact"=>true,"owned_output"=>true,
        "fixture_and_oracle_ms"=>1000fixture.time,"fixture_and_oracle_compile_ms"=>1000fixture.compile_time,
        "first_episode_ms"=>1000first_episode.time,"first_episode_compile_ms"=>1000first_episode.compile_time,
        "first_episode_bytes"=>first_episode.bytes,"fixture_bytes"=>Base.summarysize(state),
        "owned_output_bytes"=>Base.summarysize(first_episode.value),"worker_peak_rss_bytes"=>Sys.maxrss(),
        "certifications"=>diagnostic.certifications,"refusals"=>diagnostic.refusals,"reuse"=>diagnostic.reuse,
        "counters_from_separate_untimed_replay"=>true,
        "exact_phase_values"=>[[string(k)*"="*string(v) for (k,v) in sort!(collect(a);by=Base.first)] for a in state.exact])
    worker_trace=joinpath(dirname(trace),"case-$(record.case_id)-worker-$(getpid())-diagnostics.toml")
    isfile(worker_trace) || open(io->TOML.print(io,details),worker_trace,"w")
    if isfile(trace)
        prior=TOML.parsefile(trace)
        prior["case_id"]==record.case_id && prior["source_sha256"]==expected_hash || error("N04 trace identity mismatch")
    else
        open(io->TOML.print(io,details),trace,"w")
    end
    state
end

function n04_worker_oracle(state,strategy)
    output=n04_episode(state,strategy)
    n04_resource_check(state,output) && n04_output_check(state,output)
end

function n04_write_records(path,records)
    open(path,"w") do io
        println(io,"case_id,n,exponent,horizon,metric,shape,changes,strategy")
        for r in records;println(io,join(values(r),','));end
    end
end

function n04_disk_check(root;max_bytes=64<<20)
    bytes=sum(filesize,filter(isfile,readdir(root;join=true));init=0)
    bytes<=max_bytes || error("N04 results disk budget exceeded")
    bytes
end

function n04_before_next_case(elapsed;wall_seconds=300,reserve_seconds=60)
    0<=elapsed && wall_seconds>reserve_seconds>=0 || throw(ArgumentError("invalid N04 time budget"))
    elapsed<wall_seconds-reserve_seconds
end

function n04_archive(root)
    repo=dirname(@__DIR__);files=n04_source_files()
    append!(files,["perf/n04_short_identity_tests.jl","perf/n04_short_identity_controller_tests.jl"])
    archive=joinpath(root,"n04-sources.tar.gz")
    run(Cmd(vcat(["tar","-czf",archive,"-C",repo],files)))
    bytes2hex(sha256(read(archive)))
end

function n04_campaign(root;smoke=false,case_ids=nothing,prepare_only=false,reverse_order=false)
    n04_runtime_check();condition=n04_condition()
    records=n04_grid();selected=n04_select(records;smoke,case_ids,reverse_order)
    !prepare_only && isempty(selected) && throw(ArgumentError("no implicit N04 sweep: select --smoke or --case-ids"))
    fingerprint=n04_source_hash();grid_hash=n04_grid_hash(records)
    mkpath(root)
    samples=joinpath(root,"n04-samples.csv");meta_file=joinpath(root,"n04-environment.toml")
    isfile(samples) && error("N04 observations already exist; use a new passage root")
    if isfile(meta_file)
        prior=TOML.parsefile(meta_file)
        prior["status"]=="prepared" && prior["source_sha256"]==fingerprint &&
            prior["case_ids"]==getproperty.(selected,:case_id) && prior["measurement_condition"]==condition.condition &&
            prior["interference_label"]==condition.label || error("N04 prepared context changed; use a new root")
    end
    n04_write_records(joinpath(root,"n04-replay-matrix.csv"),records)
    n04_write_records(joinpath(root,"n04-selection.csv"),selected)
    metadata=Dict{String,Any}("status"=>"prepared","grid_sha256"=>grid_hash,"grid_cases"=>length(records),
        "source_sha256"=>fingerprint,"case_ids"=>getproperty.(selected,:case_id),"reverse_order"=>reverse_order,
        "measurement_condition"=>condition.condition,"interference_label"=>condition.label,
        "automatically_verified_isolation"=>false,"julia"=>string(VERSION),"threads"=>Threads.nthreads(),
        "julia_executable"=>joinpath(Sys.BINDIR,Base.julia_exename()),"cpu"=>Sys.cpu_info()[1].model,
        "wall_seconds"=>300,"reserve_before_case_seconds"=>60,"rss_checkpoint_bytes"=>2<<30,
        "fixture_bytes_limit"=>64<<20,"output_bytes_limit"=>64<<20,"disk_bytes_limit"=>64<<20,
        "completed_case_ids"=>Int[],"failed_case_ids"=>Int[],"under_sampled_case_ids"=>Int[],"current_case_id"=>0,
        "samples_requested"=>15,"sample_seconds"=>5.0,"evals"=>1,
        "budget_enforcement"=>"checkpoints; external process-tree supervisor required",
        "required_number_contract"=>"Rational{BigInt}","output_contract"=>"full owned sparse multivector at each horizon step")
    save_metadata()=open(io->TOML.print(io,metadata),meta_file,"w")
    save_metadata();n04_disk_check(root)
    prepare_only && return metadata
    isdefined(@__MODULE__,:run_suite) || error("load PerfChecker before calling the measurement API")
    get(ENV,"OPENBLAS_NUM_THREADS","")=="1" && get(ENV,"OMP_NUM_THREADS","")=="1" ||
        error("set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before N04 measurements")
    metadata["perfchecker"]=string(pkgversion(PerfChecker));metadata["source_archive_sha256"]=n04_archive(root)
    metadata["status"]="running";save_metadata()
    started=time()
    open(io->println(io,"case_id,n,exponent,horizon,metric,shape,changes,strategy,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations"),samples,"w")
    mktempdir(;prefix="garamon-n04-") do temporary
        for r in selected
            n04_before_next_case(time()-started) || (metadata["status"]="wall_budget_before_next_case";save_metadata();break)
            n04_source_hash()==fingerprint || error("N04 source/environment drift")
            metadata["current_case_id"]=r.case_id;save_metadata()
            entry=joinpath(temporary,"case.jl");trace=joinpath(root,"case-$(r.case_id)-diagnostics.toml")
            open(entry,"w") do io
                println(io,"include(",repr(@__FILE__),")")
                println(io,"perf_setup()=n04_worker_setup(",repr(r),",",repr(trace),",",repr(fingerprint),")")
                println(io,"perf_workload(state)=n04_episode(state,:",r.strategy,")")
                println(io,"perf_oracle(state)=n04_worker_oracle(state,:",r.strategy,")")
            end
            feature=FeatureSpec(Symbol(:n04_,r.case_id);backend=:benchmark,entrypoint=entry,
                description="N04 exact rational powers, full owned output, complete preparation cost",
                comparison_key="N04/$(r.n)/$(r.p)/$(r.H)/$(r.family)/$(r.shape)/$(r.changes)",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>5.0,:threads=>1))
            package=PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
                versions=VersionNumber[],dev_sources=String[],features=[feature])
            result=run_suite(SoftwareSuite(:n04_short_identity,[package]);profile=:quick,strict=false)
            write_suite_json(result,joinpath(root,"case-$(r.case_id)-qualification.json"))
            run=only(result.runs)
            sample_count=0
            open(samples,"a") do io
                if run.status==:pass
                    table=only(run.result.tables);sample_count=length(table.times)
                    for j in eachindex(table.times)
                        println(io,join((values(r)...,run.status,j,table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((values(r)...,run.status,"","","","",""),','))
                end
            end
            if run.status!=:pass
                push!(metadata["failed_case_ids"],r.case_id)
            elseif sample_count!=15
                push!(metadata["under_sampled_case_ids"],r.case_id)
            end
            metadata["case_$(r.case_id)_samples"]=sample_count
            push!(metadata["completed_case_ids"],r.case_id);metadata["current_case_id"]=0
            metadata["elapsed_seconds"]=time()-started;save_metadata()
            n04_source_hash()==fingerprint || error("N04 source/environment drift")
            n04_disk_check(root)
        end
    end
    metadata["elapsed_seconds"]=time()-started
    metadata["status"]=length(metadata["completed_case_ids"])==length(selected) &&
        isempty(metadata["failed_case_ids"]) && isempty(metadata["under_sampled_case_ids"]) ?
        "qualified" : "incomplete_or_failed"
    save_metadata()
    metadata["status"]=="qualified" || error("N04 incomplete or failed; raw evidence retained")
    metadata
end

function n04_main(args)
    prepare=false;smoke=false;reverse_order=false;ids=nothing;paths=String[]
    for arg in args
        if arg=="--prepare-only";prepare=true
        elseif arg=="--smoke";smoke=true
        elseif arg=="--reverse";reverse_order=true
        elseif startswith(arg,"--case-ids=")
            ids===nothing || throw(ArgumentError("duplicate case selection"))
            ids=parse.(Int,split(chopprefix(arg,"--case-ids="),','))
        elseif startswith(arg,"--");throw(ArgumentError("unknown N04 option"))
        else;push!(paths,arg)
        end
    end
    length(paths)==1 || throw(ArgumentError("one N04 output directory required"))
    metadata=n04_campaign(abspath(only(paths));smoke,case_ids=ids,prepare_only=prepare,reverse_order)
    println("N04 ",metadata["status"],": ",length(metadata["case_ids"])," selected / ",metadata["grid_cases"]," identities")
    metadata
end

if abspath(PROGRAM_FILE)==@__FILE__
    n04_main(ARGS)
end
