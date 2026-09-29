using CUDA, Garamon, PerfChecker, BenchmarkTools, Statistics, SHA, TOML, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("gpu_packed.jl")
include("cpu_threaded_packed.jl")
using .GPUPackedPrototype, .CPUThreadedPackedPrototype

const THREAD_SMOKE_DIMENSIONS=(8,65)
const THREAD_SCREEN_DIMENSIONS=(2,3,4,6,8,12,16,32,64,65,96,128)
const THREAD_SMOKE_HORIZONS=(1,32,1024,8192)
const THREAD_SCREEN_HORIZONS=(1024,8192)
const THREAD_ORDERS=((:serial,:threaded,:gpu),(:threaded,:gpu,:serial),
    (:gpu,:serial,:threaded))
const THREAD_SAMPLES=7

thread_case_id(c)="THREAD-n$(lpad(string(c.n),3,'0'))-$(c.family)-H$(c.horizon)-$(c.route)-pass$(c.passage)"

function thread_make_batch(fixture,horizon)
    a,b=first(fixture.inputs)
    plan=prepare_product(a,b)
    left=[fixture.inputs[mod1(i,4)][1] for i in 1:horizon]
    right=[fixture.inputs[mod1(i,4)][2] for i in 1:horizon]
    pack_product_batch(plan,left,right)
end

thread_serial_full(fixture,horizon)=run_packed_batch(thread_make_batch(fixture,horizon))
thread_parallel_full(fixture,horizon)=run_packed_batch_threaded(thread_make_batch(fixture,horizon))
thread_gpu_full(fixture,horizon)=gpu_complete_matrix(thread_make_batch(fixture,horizon);max_bytes=512<<20)

function thread_oracle(fixture,batch,matrix,horizon)
    size(matrix)==(length(batch.plan.output_masks),horizon) || return false
    expected=[rank_oracle(fixture,pair...,(:all,0)) for pair in fixture.inputs]
    for col in 1:horizon,(row,mask) in enumerate(batch.plan.output_masks)
        matrix[row,col]==get(expected[mod1(col,4)],mask,Int64(0)) || return false
    end
    true
end

function cpu_threaded_campaign(root;mode::Symbol=:smoke)
    mode in (:smoke,:screen) || error("threaded mode must be smoke or screen")
    VERSION.major==1 && VERSION.minor==13 || error("threaded pilot requires Julia 1.13")
    Threads.nthreads()==4 || error("threaded pilot requires exactly four Julia threads")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated label must be empty") :
        !isempty(label) || error("exploratory label required")
    dimensions=mode==:smoke ? THREAD_SMOKE_DIMENSIONS : THREAD_SCREEN_DIMENSIONS
    horizons=mode==:smoke ? THREAD_SMOKE_HORIZONS : THREAD_SCREEN_HORIZONS
    families=mode==:smoke ? (:higher_rank,) : (:low_rank_high_grade,:higher_rank)
    cases=[(;n,family,horizon,route,passage) for (passage,order) in enumerate(THREAD_ORDERS)
        for n in dimensions for family in families for horizon in horizons for route in order]
    root=abspath(root);ispath(root) && error("choose a fresh output directory")
    mkpath(root)
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"cpu_threaded_packed.jl"),
        joinpath(@__DIR__,"gpu_packed.jl"),joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank_cases.jl"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Project.toml"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Manifest.toml")],
        [joinpath(dirname(@__DIR__),"src",file) for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    original=fingerprint();started=time();completed=String[]
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running","technique"=>"CPU-packed-column-threading-v1",
        "mode"=>string(mode),"condition"=>condition,"interference_label"=>label,
        "isolation_certified"=>false,"julia"=>string(VERSION),
        "cuda_jl"=>string(pkgversion(CUDA)),"perfchecker"=>string(pkgversion(PerfChecker)),
        "cpu"=>Sys.CPU_NAME,"gpu"=>string(CUDA.device()),"gpu_before"=>smi(),
        "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "dimensions"=>collect(dimensions),"horizons"=>collect(horizons),
        "families"=>collect(string.(families)),"routes"=>["serial","threaded","gpu"],
        "passes"=>length(THREAD_ORDERS),"case_ids"=>thread_case_id.(cases),
        "case_count"=>length(cases),"completed_case_ids"=>completed,
        "samples"=>THREAD_SAMPLES,"source_files"=>sourcefiles,
        "source_sha256"=>original,"source_unchanged"=>false,
        "argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false),
        "contract"=>"same ordered packed product paths; serial CPU, static four-thread CPU by independent columns, or synchronized GPU; new owned Float64 host matrix",
        "phase_contract"=>"build=plan+packing (+GPU upload); hot=prepared input to owned host matrix; episode=whole operation directly timed",
        "gpu_budget_bytes"=>512<<20,"rss_budget_bytes"=>4<<30,
        "wall_budget_seconds"=>(mode==:smoke ? 1200 : 2400))
    manifest=joinpath(root,"manifest.toml")
    open(io->TOML.print(io,metadata),manifest,"w")
    samplesfile=joinpath(root,"samples.csv")
    write(samplesfile,"case_id,phase,sample,time_ns,gc_time_ns,cpu_allocated_bytes,cpu_allocations\n")
    firstfile=joinpath(root,"first-observations.csv")
    write(firstfile,"case_id,phase,time_ns,compile_ns,cpu_allocated_bytes,gc_time_ns\n")
    function recordfirst(id,timing)
        open(firstfile,"a") do io
            println(io,join((id,"first_episode_in_case",timing.time*1e9,
                timing.compile_time*1e9,timing.bytes,timing.gctime*1e9),','))
        end
    end
    function recordsamples(id,phase,trial)
        length(trial.times)==THREAD_SAMPLES || error("insufficient samples")
        open(samplesfile,"a") do io
            for i in eachindex(trial.times)
                println(io,join((id,phase,i,trial.times[i],trial.gctimes[i],
                    trial.memory,trial.allocs),','))
            end
        end
    end
    fixtures=Dict{Tuple,Any}();batches=Dict{Tuple,Any}();residents=Dict{Tuple,Any}()
    features=[FeatureSpec(Symbol("thread_"*string(i));entrypoint=@__FILE__,backend=:benchmark,
        oracle=OracleSpec(),comparison_key="CPU-threaded/$(c.n)/$(c.family)/H$(c.horizon)/pass$(c.passage)",
        options=Dict(:thread_case=>c)) for (i,c) in enumerate(cases)]
    suite=SoftwareSuite(:cpu_threaded,[PackageSuite("Garamon";source=dirname(@__DIR__),
        worker_environment=joinpath(@__DIR__,"..","..","GaramonBench","gpu"),
        versions=VersionNumber[],dev_sources=String[],features)])
    function executor(planned,config,setup,workload)
        time()-started<=metadata["wall_budget_seconds"] || error("threaded pilot wall budget exceeded")
        Sys.maxrss()<=metadata["rss_budget_bytes"] || error("threaded pilot RSS budget exceeded")
        case=planned.feature.options[:thread_case];id=thread_case_id(case)
        fixture=get!(fixtures,(case.n,case.family)) do
            compact_fixture(case.n,case.family,:positive)
        end
        batch=get!(batches,(case.n,case.family,case.horizon)) do
            thread_make_batch(fixture,case.horizon)
        end
        resident=get!(residents,(case.n,case.family,case.horizon)) do
            gpu_resident_batch(batch;max_bytes=512<<20)
        end
        route=case.route
        full()=route==:serial ? thread_serial_full(fixture,case.horizon) :
            route==:threaded ? thread_parallel_full(fixture,case.horizon) :
            thread_gpu_full(fixture,case.horizon)
        first=@timed full()
        recordfirst(id,first)
        thread_oracle(fixture,batch,first.value,case.horizon) || error("first oracle failed")
        for _ in 1:2
            thread_oracle(fixture,batch,run_packed_batch(batch),case.horizon) || error("serial warm oracle failed")
            thread_oracle(fixture,batch,run_packed_batch_threaded(batch),case.horizon) || error("threaded warm oracle failed")
            thread_oracle(fixture,batch,gpu_owned_matrix(resident),case.horizon) || error("GPU warm oracle failed")
        end
        build=route==:gpu ?
            (@benchmark gpu_resident_batch(thread_make_batch($fixture,$(case.horizon));max_bytes=512<<20) samples=THREAD_SAMPLES evals=1 seconds=0.2) :
            (@benchmark thread_make_batch($fixture,$(case.horizon)) samples=THREAD_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:build,build)
        hot=route==:serial ?
            (@benchmark run_packed_batch($batch) samples=THREAD_SAMPLES evals=1 seconds=0.2) :
            route==:threaded ?
            (@benchmark run_packed_batch_threaded($batch) samples=THREAD_SAMPLES evals=1 seconds=0.2) :
            (@benchmark gpu_owned_matrix($resident) samples=THREAD_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:hot,hot)
        episode=route==:serial ?
            (@benchmark thread_serial_full($fixture,$(case.horizon)) samples=THREAD_SAMPLES evals=1 seconds=0.2) :
            route==:threaded ?
            (@benchmark thread_parallel_full($fixture,$(case.horizon)) samples=THREAD_SAMPLES evals=1 seconds=0.2) :
            (@benchmark thread_gpu_full($fixture,$(case.horizon)) samples=THREAD_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:episode,episode)
        observed=full()
        thread_oracle(fixture,batch,observed,case.horizon) || error("post-sample oracle failed")
        episode.memory<=512<<20 || error("CPU allocation budget exceeded")
        push!(completed,id)
        open(io->TOML.print(io,metadata),manifest,"w")
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,
            [:CPU,:GPU,Symbol(condition)],[PerfChecker.PackageSpec(name="Garamon")],
            [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"independent Int64 ordered-word oracle for all owned output cells"),
                "case_id"=>id,"source_sha256"=>original,"route"=>string(route))])
    end
    try
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(root,"qualification.json"))
        metadata["source_unchanged"]=fingerprint()==original
        metadata["status"]=suite_passed(result) && metadata["source_unchanged"] &&
            length(completed)==length(cases) ? "validated" : "failed"
        metadata["verdict"]=string(suite_verdict(result))
        metadata["status"]=="validated" || error("threaded pilot qualification failed")
    finally
        metadata["elapsed_seconds"]=time()-started
        metadata["rss_bytes"]=Sys.maxrss()
        metadata["gpu_after"]=smi()
        metadata["not_qualified_case_ids"]=setdiff(metadata["case_ids"],completed)
        open(io->TOML.print(io,metadata),manifest,"w")
    end
    root
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (1,2) || error("usage: cpu_threaded_compare.jl NEW_OUTPUT_DIRECTORY [smoke|screen]")
    BLAS.set_num_threads(1)
    println(cpu_threaded_campaign(ARGS[1];mode=length(ARGS)==2 ? Symbol(ARGS[2]) : :smoke))
end
