using CUDA, Garamon, PerfChecker, BenchmarkTools, Statistics, SHA, TOML, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("gpu_packed.jl")
using .GPUPackedPrototype

const GPU_SCREEN_DIMENSIONS=(2,3,4,6,8,12,16,32,64,65,96,128)
const GPU_FAMILIES=(:low_rank_high_grade,:higher_rank)
const GPU_HORIZONS=(1,32,1024,8192)
const GPU_ROUTES=(:cpu,:gpu)
const GPU_SAMPLES=7

function gpu_case_id(case)
    "GPU-n$(lpad(string(case.n),3,'0'))-$(case.family)-$(case.signature)-H$(case.horizon)-$(case.route)-pass$(case.passage)"
end

function gpu_make_batch(fixture,horizon)
    a,b=first(fixture.inputs)
    plan=prepare_product(a,b)
    left=[fixture.inputs[mod1(i,4)][1] for i in 1:horizon]
    right=[fixture.inputs[mod1(i,4)][2] for i in 1:horizon]
    pack_product_batch(plan,left,right)
end

gpu_full(fixture,horizon)=gpu_complete_matrix(gpu_make_batch(fixture,horizon);max_bytes=512<<20)
cpu_full(fixture,horizon)=run_packed_batch(gpu_make_batch(fixture,horizon))

function gpu_exact_oracle(fixture,batch,matrix,horizon)
    size(matrix)==(length(batch.plan.output_masks),horizon) || return false
    expected=[rank_oracle(fixture,pair...,(:all,0)) for pair in fixture.inputs]
    for column in 1:horizon,(row,mask) in enumerate(batch.plan.output_masks)
        matrix[row,column]==get(expected[mod1(column,4)],mask,Int64(0)) || return false
    end
    true
end

function gpu_packed_campaign(output;mode::Symbol=:smoke)
    mode in (:smoke,:screen) || error("GPU packed mode must be smoke or screen")
    VERSION.major==1 && VERSION.minor==13 || error("GPU packed pilot requires Julia 1.13")
    Threads.nthreads()==1 || error("GPU packed pilot requires one Julia thread")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare GPU measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated GPU run requires empty label") :
        !isempty(label) || error("exploratory GPU run requires interference label")
    dimensions=mode==:smoke ? (8,65) : GPU_SCREEN_DIMENSIONS
    horizons=mode==:smoke ? (1,1024,8192) : GPU_HORIZONS
    families=mode==:smoke ? (:higher_rank,) : GPU_FAMILIES
    cases=NamedTuple[]
    for passage in 1:2,n in dimensions,family in families,horizon in horizons,
        route in (passage==1 ? GPU_ROUTES : reverse(GPU_ROUTES))
        push!(cases,(;n,family,signature=:positive,horizon,route,passage))
    end
    output=abspath(output);ispath(output) && error("use a fresh GPU output directory")
    mkpath(output)
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"gpu_packed.jl"),
        joinpath(@__DIR__,"binary_rank_compact_cases.jl"),joinpath(@__DIR__,"binary_rank_cases.jl"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Project.toml"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Manifest.toml")],
        [joinpath(dirname(@__DIR__),"src",file) for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    original=fingerprint();completed=String[];started=time()
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running","technique"=>"GPU-packed-output-CSR-v1",
        "mode"=>string(mode),"julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "perfchecker"=>string(pkgversion(PerfChecker)),"condition"=>condition,
        "interference_label"=>label,"isolation_certified"=>false,
        "isolation_status"=>"operator declaration; desktop GPU and CPU services may remain active",
        "gpu_before"=>smi(),"cpu"=>Sys.CPU_NAME,"gpu"=>string(CUDA.device()),
        "threads"=>Threads.nthreads(),"dimensions"=>collect(dimensions),
        "horizons"=>collect(horizons),"families"=>collect(string.(families)),
        "case_ids"=>gpu_case_id.(cases),"case_count"=>length(cases),
        "completed_case_ids"=>completed,"samples"=>GPU_SAMPLES,
        "source_sha256"=>original,"source_files"=>sourcefiles,"source_unchanged"=>false,
        "argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false),
        "contract"=>"same prepared path plan, ordered Float64 packed columns, host-owned dense output; independent Int64 word oracle",
        "phase_contract"=>"build=plan plus packing (+ GPU upload when selected); hot=prepared input to host-owned matrix; episode=direct plan plus packing plus full route to host-owned matrix",
        "gpu_budget_bytes"=>512<<20,"rss_budget_bytes"=>4<<30,
        "wall_budget_seconds"=>(mode==:smoke ? 600 : 2400))
    manifest=joinpath(output,"manifest.toml")
    open(io->TOML.print(io,metadata),manifest,"w")
    samplesfile=joinpath(output,"samples.csv")
    write(samplesfile,"case_id,phase,sample,time_ns,gc_time_ns,cpu_allocated_bytes,cpu_allocations\n")
    firstfile=joinpath(output,"first-observations.csv")
    write(firstfile,"case_id,phase,time_ns,compile_ns,cpu_allocated_bytes,gc_time_ns\n")
    function recordfirst(id,phase,timing)
        open(firstfile,"a") do io
            println(io,join((id,phase,timing.time*1e9,timing.compile_time*1e9,
                timing.bytes,timing.gctime*1e9),','))
        end
    end
    function recordsamples(id,phase,trial)
        length(trial.times)==GPU_SAMPLES || error("insufficient GPU/CPU samples")
        open(samplesfile,"a") do io
            for i in eachindex(trial.times)
                println(io,join((id,phase,i,trial.times[i],trial.gctimes[i],
                    trial.memory,trial.allocs),','))
            end
        end
    end
    fixtures=Dict{Tuple,Any}();batches=Dict{Tuple,Any}();residents=Dict{Tuple,Any}()
    features=[FeatureSpec(Symbol("gpu_"*string(i));entrypoint=@__FILE__,backend=:benchmark,
        oracle=OracleSpec(),comparison_key="GPU/packed/$(c.n)/$(c.family)/H$(c.horizon)/pass$(c.passage)",
        options=Dict(:gpu_case=>c)) for (i,c) in enumerate(cases)]
    suite=SoftwareSuite(:gpu_packed,[PackageSuite("Garamon";source=dirname(@__DIR__),
        worker_environment=joinpath(@__DIR__,"..","..","GaramonBench","gpu"),
        versions=VersionNumber[],dev_sources=String[],features)])
    function executor(planned,config,setup,workload)
        time()-started<=metadata["wall_budget_seconds"] || error("GPU wall budget exceeded")
        Sys.maxrss()<=metadata["rss_budget_bytes"] || error("GPU RSS budget exceeded")
        case=planned.feature.options[:gpu_case];id=gpu_case_id(case)
        fixture=get!(fixtures,(case.n,case.family,case.signature)) do
            compact_fixture(case.n,case.family,case.signature)
        end
        batch=get!(batches,(case.n,case.family,case.signature,case.horizon)) do
            gpu_make_batch(fixture,case.horizon)
        end
        resident=get!(residents,(case.n,case.family,case.signature,case.horizon)) do
            gpu_resident_batch(batch;max_bytes=metadata["gpu_budget_bytes"])
        end
        route=case.route
        first=@timed route==:cpu ? cpu_full(fixture,case.horizon) : gpu_full(fixture,case.horizon)
        recordfirst(id,:first_episode_in_case,first)
        gpu_exact_oracle(fixture,batch,first.value,case.horizon) || error("GPU/CPU first oracle failed")
        # Warm both implementations irrespective of route order before samples.
        for _ in 1:2
            gpu_exact_oracle(fixture,batch,run_packed_batch(batch),case.horizon) || error("CPU warm oracle failed")
            gpu_exact_oracle(fixture,batch,gpu_owned_matrix(resident),case.horizon) || error("GPU warm oracle failed")
        end
        build=route==:cpu ?
            (@benchmark gpu_make_batch($fixture,$(case.horizon)) samples=GPU_SAMPLES evals=1 seconds=0.2) :
            (@benchmark gpu_resident_batch(gpu_make_batch($fixture,$(case.horizon));max_bytes=512<<20) samples=GPU_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:build,build)
        hot=route==:cpu ?
            (@benchmark run_packed_batch($batch) samples=GPU_SAMPLES evals=1 seconds=0.2) :
            (@benchmark gpu_owned_matrix($resident) samples=GPU_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:hot,hot)
        episode=route==:cpu ?
            (@benchmark cpu_full($fixture,$(case.horizon)) samples=GPU_SAMPLES evals=1 seconds=0.2) :
            (@benchmark gpu_full($fixture,$(case.horizon)) samples=GPU_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:episode,episode)
        if route==:gpu
            kernel=@benchmark gpu_run!($resident) samples=GPU_SAMPLES evals=1 seconds=0.2
            recordsamples(id,:device_kernel_reused_output, kernel)
        end
        observed=route==:cpu ? cpu_full(fixture,case.horizon) : gpu_full(fixture,case.horizon)
        gpu_exact_oracle(fixture,batch,observed,case.horizon) || error("GPU/CPU post-sample oracle failed")
        episode.memory<=512<<20 || error("GPU/CPU episode CPU allocation budget exceeded")
        push!(completed,id)
        open(io->TOML.print(io,metadata),manifest,"w")
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,
            [:GPU,Symbol(condition),:host_owned_episode],[PerfChecker.PackageSpec(name="Garamon")],
            [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"independent Int64 ordered-word oracle for all host-owned output cells"),
                "case_id"=>id,"source_sha256"=>original,"measurement_condition"=>condition,
                "interference_label"=>label,"route"=>string(route),
                "device"=>string(CUDA.device()))])
    end
    try
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(output,"qualification.json"))
        metadata["source_unchanged"]=fingerprint()==original
        metadata["status"]=suite_passed(result) && metadata["source_unchanged"] &&
            length(completed)==length(cases) ? "validated" : "failed"
        metadata["verdict"]=string(suite_verdict(result))
        metadata["status"]=="validated" || error("GPU packed qualification failed")
    finally
        metadata["elapsed_seconds"]=time()-started
        metadata["rss_bytes"]=Sys.maxrss()
        metadata["gpu_after"]=smi()
        metadata["not_qualified_case_ids"]=setdiff(metadata["case_ids"],completed)
        open(io->TOML.print(io,metadata),manifest,"w")
    end
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (1,2) || error("usage: gpu_packed_compare.jl NEW_OUTPUT_DIRECTORY [smoke|screen]")
    BLAS.set_num_threads(1)
    println(gpu_packed_campaign(ARGS[1];mode=length(ARGS)==2 ? Symbol(ARGS[2]) : :smoke))
end
