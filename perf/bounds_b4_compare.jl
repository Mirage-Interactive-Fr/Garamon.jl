using PerfChecker, BenchmarkTools, Statistics, SHA, TOML, LinearAlgebra
push!(LOAD_PATH,dirname(@__DIR__))
include("binary_rank_compact_cases.jl")
include("bounds_b4.jl")
using .BoundsB4Prototype

const B4_DIMENSIONS=(8,)
const B4_SCREEN_DIMENSIONS=(2,3,4,6,8,12,16,32,64,65,96,128)
const B4_FAMILIES=(:low_rank_high_grade,:higher_rank)
const B4_HORIZONS=(1,1024)
const B4_SCREEN_HORIZONS=(1,32,1024)
const B4_ROUTES=(:checked,:inbounds)
const B4_SAMPLES=7

function b4_case_id(case)
    "B4-n$(lpad(string(case.n),3,'0'))-$(case.family)-$(case.signature)-H$(case.horizon)-$(case.route)-pass$(case.passage)"
end

b4_build(fixture)=b4_workspace(first(fixture.inputs)...)

function b4_batch(fixture,workspace,route,horizon)
    outputs=[b4_product!(workspace,fixture.inputs[mod1(i,4)]...;variant=route) for i in 1:horizon]
    (;outputs,checksum=sum((sum(values(output.values);init=0.0) for output in outputs);init=0.0))
end

b4_episode(fixture,route,horizon)=b4_batch(fixture,b4_build(fixture),route,horizon)
b4_correct(fixture,result,horizon)=rank_owned_correct(fixture,result,(:all,0),horizon)

function b4_benchmark(output;mode::Symbol=:smoke)
    mode in (:smoke,:screen,:focus) || error("B4 mode must be smoke, screen or focus")
    VERSION.major==1 && VERSION.minor==13 || error("B4 benchmark requires Julia 1.13")
    Threads.nthreads()==1 || error("B4 benchmark requires one Julia thread")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("exploratory_interference","isolated") ||
        error("B4 pilot requires an explicit measurement condition")
    if condition=="exploratory_interference"
        !isempty(label) || error("B4 exploratory run requires an interference label")
    else
        isempty(label) || error("B4 isolated declaration requires an empty interference label")
    end
    options=Base.JLOptions()
    cpu=unsafe_string(options.cpu_target)
    options.check_bounds in (0,1) && options.opt_level==2 && cpu=="native" &&
        options.use_pkgimages==0 || error("B4 benchmark requires auto or yes bounds, O2/native and disabled package images")
    output=abspath(output);ispath(output) && error("use a fresh B4 output directory")
    mkpath(output)
    sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"bounds_b4.jl"),
        joinpath(@__DIR__,"binary_rank_compact.jl"),joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank.jl"),joinpath(@__DIR__,"binary_rank_cases.jl")],
        [joinpath(dirname(@__DIR__),"src",file) for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
    fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
    original=fingerprint()
    cases=NamedTuple[]
    dimensions=mode==:smoke ? B4_DIMENSIONS : mode==:focus ? (12,) : B4_SCREEN_DIMENSIONS
    horizons=mode==:focus ? (1024,) : mode==:smoke ? B4_HORIZONS : B4_SCREEN_HORIZONS
    families=mode==:focus ? (:higher_rank,) : B4_FAMILIES
    first_route=Symbol(get(ENV,"B4_FIRST_ROUTE","checked"))
    first_route in B4_ROUTES || error("B4_FIRST_ROUTE must be checked or inbounds")
    first_order=first_route==:checked ? B4_ROUTES : reverse(B4_ROUTES)
    for passage in 1:2, n in dimensions, family in families,
        horizon in horizons, route in (passage==1 ? first_order : reverse(first_order))
        push!(cases,(;n,family,signature=:positive,horizon,route,passage))
    end
    completed=String[];started=time()
    metadata=Dict{String,Any}("status"=>"running","technique"=>"B4-compact-bounds-v1",
        "mode"=>string(mode),"dimensions"=>collect(dimensions),"horizons"=>collect(horizons),
        "families"=>collect(string.(families)),"first_route"=>string(first_route),
        "case_count"=>length(cases),
        "julia"=>string(VERSION),"perfchecker"=>string(pkgversion(PerfChecker)),
        "condition"=>condition,"interference_label"=>label,"isolation_certified"=>false,
        "isolation_status"=>"operator declaration and process snapshot; desktop services may remain active",
        "cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),"check_bounds"=>Int(options.check_bounds),
        "opt_level"=>Int(options.opt_level),"cpu_target"=>cpu,
        "argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false),
        "source_sha256"=>original,"source_files"=>sourcefiles,"source_unchanged"=>false,
        "case_ids"=>b4_case_id.(cases),"completed_case_ids"=>completed,
        "phase_contract"=>"build only, H products on prepared workspace, directly measured build plus H owned outputs",
        "ownership"=>"all H outputs retained, independent and verified with ordered-word Int64 oracle",
        "samples"=>B4_SAMPLES,"wall_budget_seconds"=>(mode==:screen ? 900 : 600),
        "rss_budget_bytes"=>2<<30,"episode_allocation_budget_bytes"=>512<<20)
    manifest=joinpath(output,"manifest.toml")
    open(io->TOML.print(io,metadata),manifest,"w")
    samplefile=joinpath(output,"samples.csv")
    write(samplefile,"case_id,phase,sample,time_ns,gc_time_ns,allocated_bytes,allocations\n")
    firstfile=joinpath(output,"first-observations.csv")
    write(firstfile,"case_id,phase,time_ns,compile_ns,allocated_bytes,gc_time_ns\n")
    function firstrow(id,phase,timing)
        open(firstfile,"a") do io
            println(io,join((id,phase,timing.time*1e9,timing.compile_time*1e9,timing.bytes,timing.gctime*1e9),','))
        end
    end
    function samples(id,phase,trial)
        length(trial.times)==B4_SAMPLES || error("B4 insufficient samples")
        open(samplefile,"a") do io
            for i in eachindex(trial.times)
                println(io,join((id,phase,i,trial.times[i],trial.gctimes[i],trial.memory,trial.allocs),','))
            end
        end
    end
    fixtures=Dict{Tuple,Any}();workspaces=Dict{Tuple,Any}()
    features=[FeatureSpec(Symbol("b4_"*string(i));entrypoint=@__FILE__,backend=:benchmark,oracle=OracleSpec(),
        comparison_key="B4/owned/$(c.n)/$(c.family)/H$(c.horizon)/pass$(c.passage)",
        options=Dict(:b4_case=>c)) for (i,c) in enumerate(cases)]
    suite=SoftwareSuite(:b4_compact_bounds,[PackageSuite("Garamon";source=dirname(@__DIR__),
        worker_environment=joinpath(@__DIR__,"runner"),versions=VersionNumber[],dev_sources=String[],features)])
    function executor(planned,config,setup,workload)
        time()-started<=metadata["wall_budget_seconds"] || error("B4 wall budget exceeded")
        Sys.maxrss()<=2<<30 || error("B4 RSS budget exceeded")
        case=planned.feature.options[:b4_case];id=b4_case_id(case)
        fixture=get!(fixtures,(case.n,case.family,case.signature)) do
            compact_fixture(case.n,case.family,case.signature)
        end
        workspace=get!(workspaces,(case.n,case.family,case.signature,case.route)) do
            timing=@timed b4_build(fixture);firstrow(id,:build,timing);timing.value
        end
        first=@timed b4_batch(fixture,workspace,case.route,case.horizon)
        firstrow(id,:first_batch_in_case,first)
        b4_correct(fixture,first.value,case.horizon) || error("B4 first-output oracle failed")
        build=@benchmark b4_build($fixture) samples=B4_SAMPLES evals=1 seconds=0.2
        samples(id,:build,build)
        hot=@benchmark b4_batch($fixture,$workspace,$(case.route),$(case.horizon)) samples=B4_SAMPLES evals=1 seconds=0.2
        samples(id,:hot,hot)
        b4_correct(fixture,b4_episode(fixture,case.route,case.horizon),case.horizon) || error("B4 episode oracle failed")
        episode=@benchmark b4_episode($fixture,$(case.route),$(case.horizon)) samples=B4_SAMPLES evals=1 seconds=0.2
        samples(id,:episode,episode)
        episode.memory<=512<<20 || error("B4 episode allocation budget exceeded")
        b4_correct(fixture,b4_batch(fixture,workspace,case.route,case.horizon),case.horizon) &&
            b4_correct(fixture,b4_episode(fixture,case.route,case.horizon),case.horizon) ||
            error("B4 post-sample oracle failed")
        push!(completed,id)
        open(io->TOML.print(io,metadata),manifest,"w")
        PerfChecker.CheckerResult([PerfChecker.to_table(episode)],nothing,
            [:B4,Symbol(condition),:complete_episode],[PerfChecker.PackageSpec(name="Garamon")],
            [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                "message"=>"independent Int64 ordered-word oracle and all H owned outputs"),
                "case_id"=>id,"source_sha256"=>original,"measurement_condition"=>condition,
                "interference_label"=>label,"effective_options"=>Dict("bounds"=>Int(options.check_bounds),
                    "optimization"=>Int(options.opt_level),"cpu_target"=>cpu))])
    end
    try
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(output,"qualification.json"))
        metadata["source_unchanged"]=fingerprint()==original
        metadata["status"]=suite_passed(result) && metadata["source_unchanged"] &&
            length(completed)==length(cases) ? "validated" : "failed"
        metadata["verdict"]=string(suite_verdict(result))
        metadata["status"]=="validated" || error("B4 qualification failed")
    finally
        metadata["elapsed_seconds"]=time()-started
        metadata["rss_bytes"]=Sys.maxrss()
        metadata["not_qualified_case_ids"]=setdiff(metadata["case_ids"],completed)
        open(io->TOML.print(io,metadata),manifest,"w")
    end
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (1,2) || error("usage: bounds_b4_compare.jl NEW_OUTPUT_DIRECTORY [smoke|screen|focus]")
    BLAS.set_num_threads(1)
    println(b4_benchmark(ARGS[1];mode=length(ARGS)==2 ? Symbol(ARGS[2]) : :smoke))
end
