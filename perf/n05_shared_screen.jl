# Selective follow-up to the four-fixture K3 smoke. Separate source/version.
using SHA,TOML,Statistics
if abspath(PROGRAM_FILE)==@__FILE__
    using PerfChecker
end
include("n05_shared_compare.jl")

const N05_SCREEN_DIMENSIONS=(2,3,4,5,8,12,32,64,65,128,129)
const N05_SCREEN_EXPLORATORY_ROUTES=(:independent,:shared_workspace,:shared_cached)

function n05_screen_routes(condition,requested=nothing)
    exploratory=condition.condition=="exploratory_interference"
    routes=requested===nothing ? (exploratory ? (:independent,:shared_workspace) : N05_SHARED_ROUTES) : Tuple(requested)
    isempty(routes) && throw(ArgumentError("at least one screen route required"))
    length(unique(routes))==length(routes) || throw(ArgumentError("duplicate screen route"))
    all(in(N05_SHARED_ROUTES),routes) || throw(ArgumentError("unknown screen route"))
    :independent in routes || throw(ArgumentError("independent control is required"))
    :shared_workspace in routes || throw(ArgumentError("shared workspace is required"))
    exploratory && !all(in(N05_SCREEN_EXPLORATORY_ROUTES),routes) &&
        throw(ArgumentError("exploratory screen permits independent, shared_workspace and optional shared_cached only"))
    Tuple(filter(in(routes),N05_SHARED_ROUTES))
end

# A terminated batch retains its entire reservation. A clean exit records actual
# elapsed time. This is a sequential launch budget, not a concurrent job scheduler.
function n05_screen_budget(root,condition,routes,fingerprint;prepare_only=false)
    exploratory=condition.condition=="exploratory_interference"
    charged=0.0
    for name in readdir(root;join=true)
        isdir(name) && startswith(basename(name),"n05-shared-screen-batch-") || continue
        path=joinpath(name,"n05-shared-screen-environment.toml")
        isfile(path) || continue
        prior=TOML.parsefile(path)
        prior["measurement_condition"]==condition.condition && prior["interference_label"]==condition.label &&
            prior["source_sha256"]==fingerprint && prior["selected_routes"]==collect(string.(routes)) ||
            error("screen root mixes sources, conditions or selected routes; choose a fresh root")
        charged+=get(prior,"charged_seconds",0.0)
    end
    total=exploratory ? 1800.0 : 22*900.0
    remaining=max(0.0,total-charged)
    !prepare_only && remaining<60 && error("screen root time budget exhausted; preserve partial results")
    (;charged,remaining,total,wall=min(exploratory ? 300.0 : 900.0,remaining))
end

function n05_screen_grid()
    records=NamedTuple[]
    for (dimension_index,n) in enumerate(N05_SCREEN_DIMENSIONS),(order_index,order) in enumerate((:forward,:reverse)),
        horizon in (1,32),changes in (:stable,:all),strategy in N05_SHARED_ROUTES
        push!(records,(;case_id=length(records)+1,batch=2(dimension_index-1)+order_index,n,
            k=n<=8 ? 4 : 8,count=4,horizon,family=:signed,sharing=:shared,changes,strategy,order))
    end
    records
end

function n05_screen_grid_hash(records=n05_screen_grid())
    bytes2hex(sha256("K3-screen-v1\n"*join((join(values(r),',') for r in records),"\n")))
end

function n05_screen_hash()
    bytes2hex(sha256(n05_shared_hash()*"\n"*read(@__FILE__,String)*"\n"*
        read(joinpath(@__DIR__,"n05_shared_screen_protocol.toml"),String)))
end

# Separate diagnostic probes of the actual constructors used at episode entry.
# They do not replace the main workload, and cannot be subtracted from its time.
function n05_screen_prepare_probe(state,strategy)
    G=state.G;U=first(state.pools)
    strategy==:independent && return nothing
    if strategy==:workspace_unshared
        return [begin
            pool=U[:,chain];plan=n05_shared_plan(state.n,length(chain),[collect(1:length(chain))])
            (;pool,workspace=n05_shared_workspace(plan,G,pool))
        end for chain in state.chains]
    end
    plan=n05_shared_plan(state.n,size(U,2),state.chains)
    strategy==:shared_allocated && return plan # Its workspace is built inside each iteration.
    (;plan,workspace=n05_shared_workspace(plan,G,U;policy=strategy==:shared_cached ? :check_inputs : :recompute))
end

function n05_screen_setup(record,strategy,trace)
    state=n05_shared_setup(record.n,record.k,record.count,record.horizon,record.family,record.sharing,record.changes,strategy,trace)
    if strategy!=:independent
        n05_screen_prepare_probe(state,strategy) # Compile this diagnostic wrapper first.
        probes=[@timed(n05_screen_prepare_probe(state,strategy)) for _ in 1:7]
        details=TOML.parsefile(trace)
        details["warm_preparation_constructor_probe_ms"]=[1000p.time for p in probes]
        details["warm_preparation_constructor_probe_bytes"]=[p.bytes for p in probes]
        details["warm_preparation_constructor_probe_compile_ms"]=[1000p.compile_time for p in probes]
        details["preparation_probe_scope"]="separate_warm_constructor_probe_not_additive_to_episode; allocated_route_plan_only"
        open(io->TOML.print(io,details),trace,"w")
    end
    state
end

function n05_screen_campaign(root;batch=nothing,prepare_only=false,routes=nothing)
    condition=n05_condition();records=n05_screen_grid();grid_hash=n05_screen_grid_hash(records)
    selected_routes=n05_screen_routes(condition,routes)
    mkpath(root)
    if batch===nothing
        prepare_only || error("choose --batch=1..22; no implicit large sweep")
        open(joinpath(root,"n05-shared-screen-matrix.csv"),"w") do io
            println(io,"case_id,batch,n,chain_length,chains,horizon,metric,sharing,changes,strategy,order")
            for r in records;println(io,join(values(r),','));end
        end
        open(io->TOML.print(io,Dict("grid_sha256"=>grid_hash,"cases"=>length(records),"batches"=>22,
            "status"=>"prepared_not_measured","source_sha256"=>n05_screen_hash())),joinpath(root,"n05-shared-screen-grid.toml"),"w")
        return println("K3 screen prepared: 440 cases, 22 batches, no timing")
    end
    1<=batch<=22 || throw(ArgumentError("screen batch must be 1..22"))
    fingerprint=n05_screen_hash()
    budget=n05_screen_budget(root,condition,selected_routes,fingerprint;prepare_only)
    selected=filter(r->r.batch==batch && r.strategy in selected_routes,records)
    first(selected).order==:reverse && reverse!(selected)
    destination=joinpath(root,"n05-shared-screen-batch-$(lpad(batch,2,'0'))")
    mkpath(destination);samples=joinpath(destination,"n05-shared-screen-samples.csv")
    isfile(samples)&&error("screen batch already measured; choose another root")
    meta_file=joinpath(destination,"n05-shared-screen-environment.toml")
    isfile(meta_file) && get(TOML.parsefile(meta_file),"charged_seconds",0.0)>0 &&
        error("screen batch already reserved; preserve its budget and partial results")
    metadata=Dict{String,Any}("grid_sha256"=>grid_hash,"source_sha256"=>fingerprint,"batch"=>batch,
        "case_ids"=>getproperty.(selected,:case_id),"measurement_condition"=>condition.condition,
        "interference_label"=>condition.label,"condition_declaration"=>condition.source,
        "concurrent_external_jobs"=>condition.concurrent_external_jobs,"status"=>"prepared",
        "completed_case_ids"=>Int[],"failed_case_ids"=>Int[],"current_case_id"=>0,
        "selected_routes"=>collect(string.(selected_routes)),"threads"=>1,"max_cases"=>length(selected),
        "wall_seconds"=>budget.wall,"root_wall_seconds"=>budget.total,"prior_charged_seconds"=>budget.charged,
        "charged_seconds"=>0.0,"rss_checkpoint_bytes"=>2<<30,
        "fixture_and_workspace_bytes"=>64<<20,"disk_bytes_per_batch"=>64<<20,
        "budget_enforcement"=>"sequential checkpoints; external timeout required for hard wall limit")
    save_metadata()=open(io->TOML.print(io,metadata),meta_file,"w")
    save_metadata();prepare_only && return println("Prepared screen batch ",batch)
    Threads.nthreads()==1 || error("screen requires one Julia thread")
    metadata["julia"]=string(VERSION);metadata["perfchecker"]=string(pkgversion(PerfChecker))
    metadata["cpu"]=Sys.cpu_info()[1].model
    metadata["controller_manifest_sha256"]=bytes2hex(sha256(read(joinpath(@__DIR__,"controller","Manifest.toml"))))
    started=time();metadata["charged_seconds"]=budget.wall;save_metadata()
    open(io->println(io,"case_id,batch,n,chain_length,chains,horizon,metric,sharing,changes,strategy,order,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations"),samples,"w")
    mktempdir(;prefix="garamon-k3-screen-") do temporary
        for r in selected
            time()-started<budget.wall-60 || (metadata["status"]="wall_budget_before_next_case";save_metadata();break)
            entry=joinpath(temporary,"case.jl");trace=joinpath(destination,"case-$(r.case_id)-diagnostics.toml")
            metadata["current_case_id"]=r.case_id;metadata["status"]="running";save_metadata()
            open(entry,"w") do io
                println(io,"include(",repr(@__FILE__),")")
                println(io,"perf_setup()=n05_screen_setup(",repr(r),",:",r.strategy,",",repr(trace),")")
                println(io,"perf_workload(state)=n05_shared_episode(state,:",r.strategy,")")
                println(io,"perf_oracle(state)=Sys.maxrss()<=2<<30 && n05_shared_qualify(state,perf_workload(state))")
            end
            feature=FeatureSpec(Symbol(:k3_screen_,r.case_id);backend=:benchmark,entrypoint=entry,
                description="K3 selective screen, owned output, preparation included",
                comparison_key="K3/screen/$(r.n)/$(r.k)/$(r.count)/$(r.horizon)/$(r.changes)",
                state_policy=:reuse,oracle=OracleSpec(function_name=:perf_oracle),
                options=Dict(:samples=>15,:evals=>1,:seconds=>0.1,:threads=>1))
            package=PackageSuite("Garamon";source=dirname(@__DIR__),worker_environment=joinpath(@__DIR__,"runner"),
                versions=VersionNumber[],dev_sources=String[],features=[feature])
            result=run_suite(SoftwareSuite(:k3_screen,[package]);profile=:quick,strict=false)
            write_suite_json(result,joinpath(destination,"case-$(r.case_id)-qualification.json"))
            run=only(result.runs)
            open(samples,"a") do io
                if run.status==:pass
                    table=only(run.result.tables)
                    for j in eachindex(table.times)
                        println(io,join((values(r)...,run.status,j,table.times[j],table.gctimes[j],table.memory[j],table.allocs[j]),','))
                    end
                else
                    println(io,join((values(r)...,run.status,"","","","",""),','));push!(metadata["failed_case_ids"],r.case_id)
                end
            end
            push!(metadata["completed_case_ids"],r.case_id);metadata["current_case_id"]=0
            metadata["elapsed_seconds"]=time()-started;save_metadata()
            fingerprint==n05_screen_hash() || error("K3 screen sources changed")
            sum(filesize,filter(isfile,readdir(destination;join=true));init=0)<=64<<20 || error("screen disk budget")
        end
    end
    complete=length(metadata["completed_case_ids"])==length(selected)
    metadata["status"]=complete&&isempty(metadata["failed_case_ids"]) ? "qualified" : "incomplete_or_failed"
    metadata["elapsed_seconds"]=time()-started
    metadata["charged_seconds"]=metadata["elapsed_seconds"];save_metadata()
    metadata["status"]=="qualified" || error("screen incomplete or failed; partial results retained")
    println("K3 selective screen batch ",batch," qualified")
end

function n05_screen_main(args)
    batch=nothing;prepare=false;paths=String[];routes=nothing
    for argument in args
        if argument=="--prepare-only"
            prepare=true
        elseif startswith(argument,"--batch=")
            batch===nothing||error("duplicate screen batch");batch=parse(Int,chopprefix(argument,"--batch="))
        elseif startswith(argument,"--routes=")
            routes===nothing||error("duplicate screen routes")
            routes=Tuple(Symbol.(split(chopprefix(argument,"--routes="),',')))
        elseif startswith(argument,"--")
            error("unknown screen option")
        else
            push!(paths,argument)
        end
    end
    length(paths)==1 || error("one screen output directory required")
    n05_screen_campaign(abspath(only(paths));batch,prepare_only=prepare,routes)
end

if abspath(PROGRAM_FILE)==@__FILE__
    n05_screen_main(ARGS)
end
