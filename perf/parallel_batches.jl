# Run each configuration in a fresh Julia process; see parallel_batches.md.
const PARALLEL_ENTRY_NS = time_ns()
using PerfChecker, BenchmarkTools, Statistics, SHA, Distributed
push!(LOAD_PATH,dirname(@__DIR__))
include(joinpath(@__DIR__,"parallel_batches_common.jl"))

length(ARGS) in (3,4) || error("usage: parallel_batches.jl threads|processes LANES OUTPUT.csv [--smoke]")
const PARALLEL_MODE = Symbol(ARGS[1])
const PARALLEL_LANES = parse(Int,ARGS[2])
const PARALLEL_OUTPUT = abspath(ARGS[3])
const PARALLEL_SMOKE = length(ARGS)==4 && ARGS[4]=="--smoke"
PARALLEL_MODE in (:threads,:processes) || error("unknown mode")
PARALLEL_LANES in (1,2,4,8,16) || error("unsupported lane count")
Threads.nthreads(:default) == (PARALLEL_MODE==:threads ? PARALLEL_LANES : 1) || error("launch with matching --threads=N,0")
Threads.nthreads(:interactive)==0 || error("use --threads=N,0")
PARALLEL_MODE==:processes && PARALLEL_LANES>8 && error("process pool capped at 8 by aggregate memory admission")

function parallel_fingerprint()
    paths = sort(vcat([@__FILE__,joinpath(@__DIR__,"parallel_batches_common.jl"),
                       joinpath(dirname(@__DIR__),"Project.toml")],
        [joinpath(root,name) for (root,_,files) in walkdir(joinpath(dirname(@__DIR__),"src"))
         for name in files if endswith(name,".jl")]))
    return bytes2hex(sha256(join((relpath(p,dirname(@__DIR__))*":"*bytes2hex(sha256(read(p))) for p in paths),"\n")))
end

const PARALLEL_HASH = parallel_fingerprint()
const PARALLEL_PIDS = Int[]
const PARALLEL_ROWS = NamedTuple[]
const PARALLEL_CACHE = Ref{Any}(nothing)
const PARALLEL_POOL_STARTUP = Ref(0.0)
const PARALLEL_START = time()
const PARALLEL_LOAD_SECONDS = (time_ns()-PARALLEL_ENTRY_NS)/1e9
println("PARALLEL_READY")
flush(stdout)

function parallel_pool_start()
    PARALLEL_MODE==:processes || return
    # Conservative admission, not a hard OS memory limit.
    Sys.free_memory() >= (PARALLEL_LANES+1)*(512<<20) || error("pool_memory_admission")
    started=time()
    project=Base.active_project()
    append!(PARALLEL_PIDS,addprocs(PARALLEL_LANES;
        exeflags=`--startup-file=no --threads=1,0 --gcthreads=1 --project=$(dirname(project))`,
        env=["OPENBLAS_NUM_THREADS"=>"1","JULIA_NUM_THREADS"=>"1,0"]))
    common=joinpath(@__DIR__,"parallel_batches_common.jl")
    root=dirname(@__DIR__)
    for pid in PARALLEL_PIDS
        remotecall_wait(Core.eval,pid,Main,:(push!(LOAD_PATH,$root)))
        remotecall_wait(Base.include,pid,Main,common)
    end
    PARALLEL_POOL_STARTUP[]=time()-started
end

function parallel_process_batch(pids,inputs,batch,payload_mode)
    ranges=parallel_ranges(batch,length(pids))
    futures=map(eachindex(pids)) do i
        remotecall(parallel_remote_chunk,pids[i],ranges[i]...,
                   payload_mode==:resident ? nothing : inputs)
    end
    return reduce(hcat,fetch.(futures))
end

function parallel_case_workload(cache,batch,transport)
    if PARALLEL_MODE==:threads
        return parallel_thread_batch(cache.states,cache.fixture.inputs,batch)
    end
    return parallel_process_batch(PARALLEL_PIDS,cache.fixture.inputs,batch,transport)
end

function parallel_cached(n,family)
    previous=PARALLEL_CACHE[]
    previous!==nothing && previous.key==(n,family) && return previous
    # Drop previous state before preparing another family; RSS is still a peak.
    PARALLEL_CACHE[]=nothing
    GC.gc()
    prepared=@timed parallel_fixture(n,family)
    fixture=prepared.value
    localprep=@timed (PARALLEL_MODE==:threads ? [parallel_workspace(fixture) for _ in 1:PARALLEL_LANES] : nothing)
    remote=PARALLEL_MODE==:processes ? [remotecall_fetch(parallel_remote_setup,pid,n,family) for pid in PARALLEL_PIDS] : []
    cache=(;key=(n,family),fixture,states=localprep.value,fixture_seconds=prepared.time,
        fixture_bytes=prepared.bytes,workspace_seconds=localprep.time,workspace_bytes=localprep.bytes,
        remote,serialization=parallel_serialization_probe(fixture.inputs),
        retained_bytes=Base.summarysize((fixture,localprep.value)))
    PARALLEL_CACHE[]=cache
    return cache
end

function parallel_executor(planned,config,setup,workload)
    time()-PARALLEL_START < PARALLEL_LIMITS.seconds_per_configuration || error("configuration_wall_budget")
    n,family,batch,transport=planned.feature.options[:parallel_case]
    cache=parallel_cached(n,family)
    parallel_admission(cache.fixture,batch)=="admitted" || error("unadmitted case")
    first=@timed parallel_case_workload(cache,batch,transport)
    parallel_oracle(first.value,cache.fixture,batch) || error("first independent oracle")
    first.bytes<=256<<20 || error("controller_batch_allocation_budget")
    trial=@benchmark parallel_case_workload($cache,$batch,$transport) samples=11 evals=1 seconds=0.2
    parallel_oracle(parallel_case_workload(cache,batch,transport),cache.fixture,batch) || error("post independent oracle")
    remote_probes = if PARALLEL_MODE==:processes
        ranges=parallel_ranges(batch,length(PARALLEL_PIDS))
        [remotecall_fetch(parallel_remote_allocation_probe,pid,ranges[i]...,
            transport==:resident ? nothing : cache.fixture.inputs)
         for (i,pid) in enumerate(PARALLEL_PIDS)]
    else
        []
    end
    remote_rss=PARALLEL_MODE==:processes ? sum(remotecall_fetch(parallel_remote_rss,pid) for pid in PARALLEL_PIDS) : 0
    Sys.maxrss()<=PARALLEL_LIMITS.max_rss_bytes || error("controller_rss_budget")
    Sys.maxrss()+remote_rss<=PARALLEL_LIMITS.max_pool_rss_bytes || error("aggregate_peak_rss_budget")
    row=(;mode=PARALLEL_MODE,lanes=PARALLEL_LANES,dimension=n,family,batch,transport,
        status="pass",median_us=median(trial.times)/1000,p95_us=quantile(trial.times,.95)/1000,
        products_per_second=batch/(median(trial.times)/1e9),controller_allocated_bytes=trial.memory,
        controller_allocations=trial.allocs,samples=length(trial.times),
        first_batch_ms=first.time*1000,first_controller_compile_ms=first.compile_time*1000,
        fixture_ms=cache.fixture_seconds*1000,fixture_allocated_bytes=cache.fixture_bytes,
        local_workspace_ms=cache.workspace_seconds*1000,local_workspace_allocated_bytes=cache.workspace_bytes,
        remote_setup_ms=sum(r.setup_seconds for r in cache.remote;init=0.0)*1000,
        remote_first_ms=sum(r.first_seconds for r in cache.remote;init=0.0)*1000,
        remote_first_compile_ms=sum(r.first_compile_seconds for r in cache.remote;init=0.0)*1000,
        remote_first_allocated_bytes=sum(r.first_bytes for r in cache.remote;init=0),
        remote_batch_allocated_bytes=sum(r.bytes for r in remote_probes;init=0),
        remote_batch_probe_ms=sum(r.seconds for r in remote_probes;init=0.0)*1000,
        retained_controller_bytes=cache.retained_bytes,
        retained_remote_bytes=sum(r.state_bytes for r in cache.remote;init=0),
        input_serialized_bytes=cache.serialization.bytes,
        serialize_input_us=cache.serialization.encode_seconds*1e6,
        deserialize_input_us=cache.serialization.decode_seconds*1e6,
        controller_peak_rss=Sys.maxrss(),sum_remote_peak_rss=remote_rss,
        output_bytes=8length(first.value),paths=length(cache.fixture.plan.paths),
        source_and_benchmark_sha256=PARALLEL_HASH)
    push!(PARALLEL_ROWS,row)
    qualification=Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
        "message"=>"independent exact integer oracle, all output coefficients"),
        "execution"=>Dict("mode"=>string(PARALLEL_MODE),"lanes"=>PARALLEL_LANES))
    return PerfChecker.CheckerResult([PerfChecker.to_table(trial)],nothing,
        [:garamon,:parallel_batches],[PerfChecker.PackageSpec(name="Garamon")],[qualification])
end

function parallel_campaign()
    parallel_pool_start()
    families=PARALLEL_SMOKE ? ((4,:sparse8),) :
        Tuple((n,family) for n in (4,12,65,128) for family in (:sparse8,:subalgebra64))
    batches=PARALLEL_SMOKE ? (32,) : (1,32,1024)
    transports=PARALLEL_MODE==:threads ? (:shared_readonly,) : (:resident,:four_variant_payload)
    cases=[(n,family,batch,transport) for (n,family) in families for batch in batches for transport in transports]
    mktempdir() do temporary
        features=FeatureSpec[]
        for (i,case) in enumerate(cases)
            entrypoint=joinpath(temporary,"parallel_$i.jl")
            write(entrypoint,"# Custom executor owns setup, batch execution and exact oracle.\n")
            push!(features,FeatureSpec(Symbol(:parallel_,i);description="Parallel coefficient batch $case",
                backend=:benchmark,entrypoint,comparison_key="garamon/parallel/$(case[1])/$(case[2])/$(case[3])/v1",
                oracle=OracleSpec(),options=Dict(:parallel_case=>case)))
        end
        package=PackageSuite("Garamon";worker_environment=joinpath(@__DIR__,"runner"),
            source=dirname(@__DIR__),versions=VersionNumber[],dev_sources=String[],features)
        suite=SoftwareSuite(:garamon_parallel_batches,[package];description="Exact materialized coefficient batches")
        result=run_suite(suite;profile=:quick,strict=false,executor=parallel_executor)
        parallel_fingerprint()==PARALLEL_HASH || error("source changed")
        mkpath(dirname(PARALLEL_OUTPUT))
        open(PARALLEL_OUTPUT,"w") do io
            println(io,"# julia=$VERSION, benchmark_load_seconds=$PARALLEL_LOAD_SECONDS, pool_startup_import_seconds=$(PARALLEL_POOL_STARTUP[]), verdict=$(suite_verdict(result))")
            isempty(PARALLEL_ROWS) || println(io,join(keys(first(PARALLEL_ROWS)),','))
            foreach(row->println(io,join(values(row),',')),PARALLEL_ROWS)
        end
        println("Parallel PerfChecker verdict: ",suite_verdict(result),"; cases: ",length(result.runs))
        suite_passed(result) || error("parallel campaign has failed cases; CSV contains successful cases only")
    end
end

try
    parallel_campaign()
finally
    isempty(PARALLEL_PIDS) || rmprocs(PARALLEL_PIDS)
end
