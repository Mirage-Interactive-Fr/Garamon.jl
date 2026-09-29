using CUDA, Garamon, PerfChecker, BenchmarkTools, Statistics, SHA, TOML, LinearAlgebra
include("binary_rank_compact_cases.jl")
include("resident_sum.jl")
using .ResidentSumPrototype

const SUM_SMOKE_DIMENSIONS=(8,65)
const SUM_SCREEN_DIMENSIONS=(2,3,4,6,8,12,16,32,64,65,96,128)
const SUM_SMOKE_PAIRS=((32,1),(32,8),(32,32),(1024,1),(1024,8),(1024,32))
const SUM_SCREEN_PAIRS=((32,32),(1024,8),(1024,32))
const SUM_ORDERS=((:serial,:threaded,:gpu),(:threaded,:gpu,:serial),
    (:gpu,:serial,:threaded))
const SUM_FUSED_ORDERS=((:threaded,:gpu,:fused),(:gpu,:fused,:threaded),
    (:fused,:threaded,:gpu))
const SUM_SAMPLES=7

sum_case_id(c)="SUM-n$(lpad(string(c.n),3,'0'))-$(c.family)-H$(c.horizon)-R$(c.repetitions)-$(c.route)-pass$(c.passage)"

sum_serial_full(fixture,horizon,repetitions)=resident_sum_cpu(
    build_resident_sum_cpu(fixture,horizon),repetitions)
sum_threaded_full(fixture,horizon,repetitions)=resident_sum_cpu(
    build_resident_sum_cpu(fixture,horizon),repetitions;threaded=true)
sum_gpu_full(fixture,horizon,repetitions)=resident_sum_gpu(
    build_resident_sum_gpu(fixture,horizon),repetitions)
sum_fused_full(fixture,horizon,repetitions)=resident_sum_gpu_fused(
    build_resident_sum_gpu(fixture,horizon),repetitions)

function sum_expected(fixture,horizon,repetitions)
    plan=first(resident_sum_packs(fixture,1)).plan
    masks=plan.output_masks
    maps=[[rank_oracle(fixture,fixture.inputs[mod1(column+shift,4)]...,
        (:all,0)) for column in 1:4] for shift in 0:3]
    counts=ntuple(k->count(step->mod1(step,4)==k,1:repetitions),4)
    expected=Matrix{Float64}(undef,length(masks),horizon)
    for column in 1:horizon,(row,mask) in enumerate(masks)
        expected[row,column]=sum(counts[k]*get(maps[k][mod1(column,4)],mask,Int64(0))
            for k in 1:4)
    end
    expected
end

function resident_sum_campaign(root;mode::Symbol=:smoke)
    mode in (:smoke,:screen,:fused_smoke,:fused_screen,:fused_focus) ||
        error("resident sum mode must be smoke, screen, fused_smoke, fused_screen or fused_focus")
    VERSION.major==1 && VERSION.minor==13 || error("resident sum requires Julia 1.13")
    Threads.nthreads()==4 || error("resident sum pilot requires four Julia threads")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated label must be empty") :
        !isempty(label) || error("exploratory label required")
    fused=mode in (:fused_smoke,:fused_screen,:fused_focus)
    dimensions=mode in (:smoke,:fused_smoke) ? SUM_SMOKE_DIMENSIONS : SUM_SCREEN_DIMENSIONS
    families=mode==:smoke ? (:higher_rank,) : (:low_rank_high_grade,:higher_rank)
    pairs=mode==:smoke ? SUM_SMOKE_PAIRS : SUM_SCREEN_PAIRS
    orders=fused ? (mode==:fused_focus ?
        vcat(collect(SUM_FUSED_ORDERS),collect(SUM_FUSED_ORDERS)) : SUM_FUSED_ORDERS) :
        SUM_ORDERS
    cases=[(;n,family,horizon,repetitions,route,passage)
        for (passage,order) in enumerate(orders)
        for n in dimensions for family in families
        for (horizon,repetitions) in pairs for route in order]
    if mode==:fused_focus
        focus=Set(((2,:higher_rank,1024,8),(6,:low_rank_high_grade,1024,8),
            (12,:higher_rank,1024,32)))
        filter!(c->(c.n,c.family,c.horizon,c.repetitions) in focus,cases)
    end
    root=abspath(root);ispath(root) && error("choose a fresh output directory")
    mkpath(root)
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"resident_sum.jl"),
        joinpath(@__DIR__,"cpu_threaded_packed.jl"),joinpath(@__DIR__,"gpu_packed.jl"),
        joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank_cases.jl"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Project.toml"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Manifest.toml")],
        [joinpath(dirname(@__DIR__),"src",file) for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    original=fingerprint();started=time();completed=String[]
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running","technique"=>
        (fused ? "four-resident-exact-product-sum-fused-v1" : "four-resident-exact-product-sum-v1"),
        "mode"=>string(mode),
        "condition"=>condition,"interference_label"=>label,"isolation_certified"=>false,
        "julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "perfchecker"=>string(pkgversion(PerfChecker)),"cpu"=>Sys.CPU_NAME,
        "gpu"=>string(CUDA.device()),"gpu_before"=>smi(),"threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(),"dimensions"=>collect(dimensions),
        "horizon_repetition_pairs"=>[collect(pair) for pair in pairs],
        "families"=>collect(string.(families)),"routes"=>collect(string.(first(orders))),
        "passes"=>length(orders),"case_ids"=>sum_case_id.(cases),
        "case_count"=>length(cases),"completed_case_ids"=>completed,
        "samples"=>SUM_SAMPLES,"source_files"=>sourcefiles,
        "source_sha256"=>original,"source_unchanged"=>false,
        "argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false),
        "contract"=>"sum every contribution of R products over four distinct resident packed operand sets; same ordered paths and final owned Float64 host matrix",
        "phase_contract"=>"build=plan+four packs (+GPU upload); hot=resident inputs to one final host-owned sum; episode=build+full sum directly timed",
        "gpu_budget_bytes"=>512<<20,"rss_budget_bytes"=>4<<30,
        "wall_budget_seconds"=>(mode in (:smoke,:fused_smoke,:fused_focus) ? 1800 : 3600))
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
        length(trial.times)==SUM_SAMPLES || error("insufficient samples")
        open(samplesfile,"a") do io
            for i in eachindex(trial.times)
                println(io,join((id,phase,i,trial.times[i],trial.gctimes[i],
                    trial.memory,trial.allocs),','))
            end
        end
    end
    fixtures=Dict{Tuple,Any}();cpu_residents=Dict{Tuple,Any}()
    gpu_residents=Dict{Tuple,Any}();expectations=Dict{Tuple,Any}()
    features=[FeatureSpec(Symbol("sum_"*string(i));entrypoint=@__FILE__,backend=:benchmark,
        oracle=OracleSpec(),comparison_key="resident-sum/$(mode)/$(c.n)/$(c.family)/H$(c.horizon)/R$(c.repetitions)/pass$(c.passage)",
        options=Dict(:sum_case=>c)) for (i,c) in enumerate(cases)]
    suite=SoftwareSuite(:resident_sum,[PackageSuite("Garamon";source=dirname(@__DIR__),
        worker_environment=joinpath(@__DIR__,"..","..","GaramonBench","gpu"),
        versions=VersionNumber[],dev_sources=String[],features)])
    function executor(planned,config,setup,workload)
        time()-started<=metadata["wall_budget_seconds"] || error("resident sum wall budget exceeded")
        Sys.maxrss()<=metadata["rss_budget_bytes"] || error("resident sum RSS budget exceeded")
        case=planned.feature.options[:sum_case];id=sum_case_id(case)
        fixture=get!(fixtures,(case.n,case.family)) do
            compact_fixture(case.n,case.family,:positive)
        end
        key=(case.n,case.family,case.horizon)
        cpus=get!(cpu_residents,key) do
            build_resident_sum_cpu(fixture,case.horizon)
        end
        gpus=get!(gpu_residents,key) do
            build_resident_sum_gpu(fixture,case.horizon)
        end
        expected=get!(expectations,(key,case.repetitions)) do
            sum_expected(fixture,case.horizon,case.repetitions)
        end
        route=case.route
        full()=route==:serial ? sum_serial_full(fixture,case.horizon,case.repetitions) :
            route==:threaded ? sum_threaded_full(fixture,case.horizon,case.repetitions) :
            route==:gpu ? sum_gpu_full(fixture,case.horizon,case.repetitions) :
            sum_fused_full(fixture,case.horizon,case.repetitions)
        first=@timed full()
        recordfirst(id,first)
        first.value==expected || error("first integer oracle failed")
        for _ in 1:2
            resident_sum_cpu(cpus,case.repetitions)==expected || error("serial warm oracle failed")
            resident_sum_cpu(cpus,case.repetitions;threaded=true)==expected || error("threaded warm oracle failed")
            resident_sum_gpu(gpus,case.repetitions)==expected || error("GPU warm oracle failed")
            fused && resident_sum_gpu_fused(gpus,case.repetitions)!=expected &&
                error("fused GPU warm oracle failed")
        end
        build=route in (:gpu,:fused) ?
            (@benchmark build_resident_sum_gpu($fixture,$(case.horizon)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            (@benchmark build_resident_sum_cpu($fixture,$(case.horizon)) samples=SUM_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:build,build)
        hot=route==:serial ?
            (@benchmark resident_sum_cpu($cpus,$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            route==:threaded ?
            (@benchmark resident_sum_cpu($cpus,$(case.repetitions);threaded=true) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            route==:gpu ?
            (@benchmark resident_sum_gpu($gpus,$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            (@benchmark resident_sum_gpu_fused($gpus,$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:hot,hot)
        episode=route==:serial ?
            (@benchmark sum_serial_full($fixture,$(case.horizon),$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            route==:threaded ?
            (@benchmark sum_threaded_full($fixture,$(case.horizon),$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            route==:gpu ?
            (@benchmark sum_gpu_full($fixture,$(case.horizon),$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2) :
            (@benchmark sum_fused_full($fixture,$(case.horizon),$(case.repetitions)) samples=SUM_SAMPLES evals=1 seconds=0.2)
        recordsamples(id,:episode,episode)
        full()==expected || error("post-sample integer oracle failed")
        episode.memory<=512<<20 || error("CPU allocation budget exceeded")
        push!(completed,id)
        open(io->TOML.print(io,metadata),manifest,"w")
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,
            [:CPU,:GPU,:resident_sum,Symbol(condition)],[PerfChecker.PackageSpec(name="Garamon")],
            [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"every final cell equals independent exact Int64 sum of all repeated products"),
                "case_id"=>id,"source_sha256"=>original,"route"=>string(route))])
    end
    try
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(root,"qualification.json"))
        metadata["source_unchanged"]=fingerprint()==original
        metadata["status"]=suite_passed(result) && metadata["source_unchanged"] &&
            length(completed)==length(cases) ? "validated" : "failed"
        metadata["verdict"]=string(suite_verdict(result))
        metadata["status"]=="validated" || error("resident sum qualification failed")
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
    length(ARGS) in (1,2) || error("usage: resident_sum_compare.jl NEW_OUTPUT_DIRECTORY [smoke|screen|fused_smoke|fused_screen|fused_focus]")
    BLAS.set_num_threads(1)
    println(resident_sum_campaign(ARGS[1];mode=length(ARGS)==2 ? Symbol(ARGS[2]) : :smoke))
end
