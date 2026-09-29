# PerfChecker pilot for repeated exact resident sums with a reusable CUDA graph.
include("resident_sum_compare.jl")

const GRAPH_SMOKE_DIMENSIONS=(8,65)
const GRAPH_SCREEN_DIMENSIONS=(2,3,4,6,8,12,16,32,64,65,96,128)
const GRAPH_HIGHDIM_DIMENSIONS=(129,192,256,384,512,768,1024,2048,4096,8192)
const GRAPH_WORKLOADS=((32,32,32),(1024,8,32),(1024,32,8))
const GRAPH_HIGHDIM_WORKLOADS=((32,32,32),(1024,32,8))
const GRAPH_ORDERS=((:threaded,:gpu,:fused,:graph),
    (:gpu,:fused,:graph,:threaded),(:fused,:graph,:threaded,:gpu),
    (:graph,:threaded,:gpu,:fused))
const GRAPH_SAMPLES=7

graph_case_id(c)="RG-n$(lpad(string(c.n),3,'0'))-$(c.family)-H$(c.horizon)-R$(c.repetitions)-Q$(c.sessions)-$(c.route)-pass$(c.passage)"

function graph_session(fixture,horizon,repetitions,sessions,route)
    residents=route==:threaded ? build_resident_sum_cpu(fixture,horizon) :
        build_resident_sum_gpu(fixture,horizon)
    prepared=route==:graph ? build_resident_sum_graph(residents,repetitions) : nothing
    results=Vector{Matrix{Float64}}(undef,sessions)
    for i in 1:sessions
        results[i]=route==:threaded ? resident_sum_cpu(residents,repetitions;threaded=true) :
            route==:gpu ? resident_sum_gpu(residents,repetitions) :
            route==:fused ? resident_sum_gpu_fused(residents,repetitions) :
            resident_sum_gpu_graph(prepared)
    end
    results
end

function graph_campaign(root;mode::Symbol=:smoke)
    mode in (:smoke,:screen,:highdim) || error("graph mode must be smoke, screen or highdim")
    VERSION.major==1 && VERSION.minor==13 || error("CUDA graph pilot requires Julia 1.13")
    Threads.nthreads()==4 || error("CUDA graph pilot requires four Julia threads")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated label must be empty") :
        !isempty(label) || error("exploratory label required")
    dimensions=mode==:smoke ? GRAPH_SMOKE_DIMENSIONS :
        mode==:screen ? GRAPH_SCREEN_DIMENSIONS : begin
            requested=get(ENV,"GARAMONBENCH_DIMENSION","")
            isempty(requested) && error("highdim mode requires one GARAMONBENCH_DIMENSION shard")
            n=parse(Int,requested)
            n in GRAPH_HIGHDIM_DIMENSIONS || error("dimension outside declared highdim grid")
            (n,)
        end
    families=mode==:highdim ? (:higher_rank,) :
        (:low_rank_high_grade,:higher_rank)
    workloads=mode==:highdim ? GRAPH_HIGHDIM_WORKLOADS : GRAPH_WORKLOADS
    phase_seconds=mode==:highdim ? 10.0 : 0.2
    cases=[(;n,family,horizon,repetitions,sessions,route,passage)
        for (passage,order) in enumerate(GRAPH_ORDERS)
        for n in dimensions for family in families
        for (horizon,repetitions,sessions) in workloads for route in order]
    root=abspath(root);ispath(root) && error("choose a fresh output directory")
    mkpath(root)
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"resident_sum_compare.jl"),
        joinpath(@__DIR__,"resident_sum.jl"),joinpath(@__DIR__,"cpu_threaded_packed.jl"),
        joinpath(@__DIR__,"gpu_packed.jl"),joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank_cases.jl"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Project.toml"),
        joinpath(dirname(@__DIR__),"..","GaramonBench","gpu","Manifest.toml")],
        [joinpath(dirname(@__DIR__),"src",file) for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    original=fingerprint();started=time();completed=String[]
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running",
        "technique"=>"exact-resident-sum-reused-cuda-graph-v1","mode"=>string(mode),
        "condition"=>condition,"interference_label"=>label,"isolation_certified"=>false,
        "julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "perfchecker"=>string(pkgversion(PerfChecker)),"cpu"=>Sys.CPU_NAME,
        "gpu"=>string(CUDA.device()),"gpu_before"=>smi(),"threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(),"dimensions"=>collect(dimensions),
        "families"=>collect(string.(families)),
        "workloads"=>[collect(x) for x in workloads],
        "routes"=>collect(string.(first(GRAPH_ORDERS))),"passes"=>length(GRAPH_ORDERS),
        "case_ids"=>graph_case_id.(cases),"case_count"=>length(cases),
        "completed_case_ids"=>completed,"samples"=>GRAPH_SAMPLES,
        "build_hot_window_seconds"=>phase_seconds,
        "source_files"=>sourcefiles,"source_sha256"=>original,"source_unchanged"=>false,
        "argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false),
        "contract"=>"Q separately owned exact final host sums of R products over four reused resident operand sets; all routes compute every contribution",
        "phase_contract"=>"build=resident inputs plus graph capture/instantiation for graph route; hot=one owned sum; episode=build plus Q owned sums directly timed",
        "gpu_budget_bytes"=>512<<20,
        "rss_budget_bytes"=>(mode==:highdim ? 6<<30 : 4<<30),
        "metric_storage_budget_bytes"=>256<<20,
        "wall_budget_seconds"=>(mode==:smoke ? 1800 : 3600))
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
        length(trial.times)==GRAPH_SAMPLES || error("insufficient samples")
        open(samplesfile,"a") do io
            for i in eachindex(trial.times)
                println(io,join((id,phase,i,trial.times[i],trial.gctimes[i],
                    trial.memory,trial.allocs),','))
            end
        end
    end
    fixtures=Dict{Tuple,Any}();cpu_residents=Dict{Tuple,Any}()
    gpu_residents=Dict{Tuple,Any}();graphs=Dict{Tuple,Any}()
    expectations=Dict{Tuple,Any}()
    features=[FeatureSpec(Symbol("graph_"*string(i));entrypoint=@__FILE__,backend=:benchmark,
        oracle=OracleSpec(),comparison_key="resident-graph/$(c.n)/$(c.family)/H$(c.horizon)/R$(c.repetitions)/Q$(c.sessions)/pass$(c.passage)",
        options=Dict(:graph_case=>c)) for (i,c) in enumerate(cases)]
    suite=SoftwareSuite(:resident_graph,[PackageSuite("Garamon";source=dirname(@__DIR__),
        worker_environment=joinpath(@__DIR__,"..","..","GaramonBench","gpu"),
        versions=VersionNumber[],dev_sources=String[],features)])
    function executor(planned,config,setup,workload)
        time()-started<=metadata["wall_budget_seconds"] || error("graph wall budget exceeded")
        Sys.maxrss()<=metadata["rss_budget_bytes"] || error("graph RSS budget exceeded")
        case=planned.feature.options[:graph_case];id=graph_case_id(case)
        fixture=get!(fixtures,(case.n,case.family)) do
            made=compact_fixture(case.n,case.family,:positive)
            metricbytes=Base.summarysize(metric(Base.first(made.inputs)[1].algebra))
            metricbytes<=metadata["metric_storage_budget_bytes"] ||
                error("actual metric storage budget exceeded")
            metadata["metric_storage_bytes"]=metricbytes
            made
        end
        key=(case.n,case.family,case.horizon)
        cpus=get!(cpu_residents,key) do
            build_resident_sum_cpu(fixture,case.horizon)
        end
        gpus=get!(gpu_residents,key) do
            build_resident_sum_gpu(fixture,case.horizon)
        end
        graph=get!(graphs,(key,case.repetitions)) do
            build_resident_sum_graph(gpus,case.repetitions)
        end
        expected=get!(expectations,(key,case.repetitions)) do
            sum_expected(fixture,case.horizon,case.repetitions)
        end
        route=case.route
        full()=graph_session(fixture,case.horizon,case.repetitions,case.sessions,route)
        first=@timed full()
        recordfirst(id,first)
        all(==(expected),first.value) || error("first integer oracle failed")
        length(unique(objectid.(first.value)))==case.sessions ||
            error("session outputs share host storage")
        for _ in 1:2
            resident_sum_cpu(cpus,case.repetitions;threaded=true)==expected || error("CPU warm oracle failed")
            resident_sum_gpu(gpus,case.repetitions)==expected || error("GPU warm oracle failed")
            resident_sum_gpu_fused(gpus,case.repetitions)==expected || error("fused warm oracle failed")
            resident_sum_gpu_graph(graph)==expected || error("graph warm oracle failed")
        end
        build=route==:threaded ?
            (@benchmark build_resident_sum_cpu($fixture,$(case.horizon)) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds) :
            route==:graph ?
            (@benchmark build_resident_sum_graph(build_resident_sum_gpu($fixture,$(case.horizon)),$(case.repetitions)) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds) :
            (@benchmark build_resident_sum_gpu($fixture,$(case.horizon)) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds)
        recordsamples(id,:build,build)
        hot=route==:threaded ?
            (@benchmark resident_sum_cpu($cpus,$(case.repetitions);threaded=true) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds) :
            route==:gpu ?
            (@benchmark resident_sum_gpu($gpus,$(case.repetitions)) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds) :
            route==:fused ?
            (@benchmark resident_sum_gpu_fused($gpus,$(case.repetitions)) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds) :
            (@benchmark resident_sum_gpu_graph($graph) samples=GRAPH_SAMPLES evals=1 seconds=phase_seconds)
        recordsamples(id,:hot,hot)
        episode=@benchmark graph_session($fixture,$(case.horizon),$(case.repetitions),
            $(case.sessions),$route) samples=GRAPH_SAMPLES evals=1 seconds=10.0
        recordsamples(id,:episode,episode)
        all(==(expected),full()) || error("post-sample integer oracle failed")
        episode.memory<=512<<20 || error("CPU allocation budget exceeded")
        push!(completed,id)
        open(io->TOML.print(io,metadata),manifest,"w")
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,
            [:CPU,:GPU,:resident_graph,Symbol(condition)],[PerfChecker.PackageSpec(name="Garamon")],
            [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"Q fresh owned final matrices equal exact independent integer oracle"),
                "case_id"=>id,"source_sha256"=>original,"route"=>string(route))])
    end
    try
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(root,"qualification.json"))
        metadata["source_unchanged"]=fingerprint()==original
        metadata["status"]=suite_passed(result) && metadata["source_unchanged"] &&
            length(completed)==length(cases) ? "validated" : "failed"
        metadata["verdict"]=string(suite_verdict(result))
        metadata["status"]=="validated" || error("graph qualification failed")
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
    length(ARGS) in (1,2) || error("usage: resident_sum_graph_compare.jl NEW_OUTPUT_DIRECTORY [smoke|screen|highdim]")
    BLAS.set_num_threads(1)
    println(graph_campaign(ARGS[1];mode=length(ARGS)==2 ? Symbol(ARGS[2]) : :smoke))
end
