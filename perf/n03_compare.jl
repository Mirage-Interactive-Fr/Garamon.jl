using PerfChecker, BenchmarkTools, Statistics, Dates
include("n03_radical.jl")
include("n03_shards.jl")

function n03_fingerprint()
    files=[joinpath(@__DIR__,name) for name in ("n03_radical.jl","n03_compare.jl","n03_shards.jl","n03_protocol.toml")]
    bytes2hex(sha256(join((basename(path)*":"*bytes2hex(sha256(read(path))) for path in files),"\n")))
end
function n03_condition()
    condition=get(ENV,"GARAMONBENCH_CONDITION","");label=get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL","")
    condition in ("exploratory_interference","isolated") || error("declare GARAMONBENCH_CONDITION before measuring")
    condition=="exploratory_interference" && isempty(strip(label)) && error("interference label required")
    condition=="isolated" && !isempty(label) && error("isolated requires empty interference label")
    (;condition,label)
end
function n03_campaign(output;mode=:smoke,prepare_only=false,max_cases=800,reverse_order=false,shard=nothing,interval=nothing)
    started_utc=string(now(UTC))*"Z"
    cases=n03_select(mode;shard,interval,reverse_order,prepare_only)
    ispath(output) && !isempty(readdir(output)) && error("choose a fresh output directory")
    mkpath(output);matrix=n03_cases(:full)
    n03_write_cases(joinpath(output,"n03-replay-matrix.csv"),matrix)
    n03_write_cases(joinpath(output,"n03-selection.csv"),cases)
    cp(joinpath(@__DIR__,"n03_protocol.toml"),joinpath(output,"n03-protocol.toml"))
    mkpath(joinpath(output,"sources"))
    for name in ("n03_radical.jl","n03_compare.jl","n03_shards.jl","n03_protocol.toml")
        cp(joinpath(@__DIR__,name),joinpath(output,"sources",name))
    end
    if prepare_only
        open(io->TOML.print(io,Dict("started_utc"=>started_utc,"finished_utc"=>string(now(UTC))*"Z",
            "source_sha256"=>n03_fingerprint(),"grid_sha256"=>n03_grid_sha256(),"grid_version"=>N03_GRID_VERSION,
            "mode"=>string(mode),"prepared_cases"=>length(matrix),"selected_cases"=>length(cases),
            "measured_cases"=>0,"status"=>"prepared_only_not_measured")),joinpath(output,"n03-prepared.toml"),"w")
        return println("N03 prepared ",length(matrix)," identities; ",length(cases)," selected; zero measured cases")
    end
    Threads.nthreads()==1 || error("N03 solo controller requires one Julia thread")
    BLAS.set_num_threads(1);condition=n03_condition()
    length(cases)<=max_cases || error("N03 case admission budget: $(length(cases)) > $max_cases")
    fingerprint=n03_fingerprint();started=time();cache=Ref{Any}(nothing);rows=Dict{String,Any}[]
    mktempdir(;prefix="garamon-n03-") do temporary
        entry=joinpath(temporary,"fixture.jl")
        write(entry,"include("*repr(joinpath(@__DIR__,"n03_radical.jl"))*")\n")
        features=[FeatureSpec(Symbol(:n03_,c.n,:_,c.r,:_,c.s,:_,c.family,:_,c.horizon,:_,c.strategy);
            backend=:benchmark,entrypoint=entry,
            comparison_key="n03/owned-inverse/construction/$(c.n)/$(c.r)/$(c.s)/$(c.family)/$(c.horizon)",
            oracle=OracleSpec(),options=Dict(:case=>c)) for c in cases]
        function executor(planned,config,setup,workload)
            isfile(joinpath(output,"STOP")) && error("N03 cooperative stop requested; remaining cases refused")
            time()-started<=840 || error("N03 campaign admission checkpoint; 60 seconds reserved for archive")
            Sys.maxrss()<=N03_LIMITS.rss_bytes || error("N03 RSS checkpoint")
            c=planned.feature.options[:case];key=(c.n,c.r,c.s,c.family,c.horizon)
            if cache[]===nothing || cache[].key!=key
                fixture=@timed n03_fixture(c.n,c.r,c.s,c.family,c.horizon)
                cache[]=(;key,state=fixture.value,fixture_ms=fixture.time*1000,fixture_bytes=fixture.bytes)
            end
            state=cache[].state;strategy=c.strategy
            first=@timed n03_episode(state,strategy)
            n03_qualify(state,first.value) || error("independent rational coefficient qualification failed")
            first.bytes<=256<<20 || error("N03 first-episode allocation budget")
            trial=@benchmark n03_episode($state,$strategy) samples=7 evals=1 seconds=0.05
            n03_qualify(state,n03_episode(state,strategy)) || error("post-sampling qualification failed")
            maximum_error=maximum((abs(get(out,mask,0.0)-Float64(get(state.exact[1+mod(t-1,3)],mask,0)))
                for (t,out) in enumerate(first.value) for mask in union(keys(out),keys(state.exact[1+mod(t-1,3)])));init=0.0)
            input=n03_input(c.n,c.r,c.s,c.family,1)
            radical_mask=n03_radical_mask(input.g)
            quotient=Dict(k=>v for (k,v) in input.a if iszero(k&radical_mask))
            detail=strategy==:radical ? n03_inverse(input.a,input.g;diagnostics=true) : nothing
            push!(rows,Dict("case_id"=>c.case_id,"feature"=>string(planned.feature.id),"dimension"=>c.n,"radical_dimension"=>c.r,
                "active_nonradical"=>c.s,"family"=>string(c.family),"horizon"=>c.horizon,"strategy"=>string(strategy),
                "construction_included"=>true,"first_episode_ms"=>first.time*1000,"first_compile_ms"=>first.compile_time*1000,
                "first_episode_allocated_bytes"=>first.bytes,"fixture_oracle_ms"=>cache[].fixture_ms,
                "fixture_oracle_allocated_bytes"=>cache[].fixture_bytes,"fixture_retained_bytes"=>Base.summarysize(state),
                "input_support"=>length(input.a),"output_support"=>length(first.value[1]),
                "quotient_matrix_order"=>length(n03_basis(quotient)),"full_active_matrix_order"=>length(n03_basis(input.a)),
                "series_terms"=>(detail===nothing ? 0 : detail.series_terms),"series_diagnostics_executed"=>detail!==nothing,"maximum_coefficient_absolute_error"=>maximum_error,
                "median_ns"=>median(trial.times),"p95_ns"=>quantile(trial.times,.95),
                "allocated_bytes"=>trial.memory,"allocations"=>trial.allocs,"samples"=>length(trial.times)))
            checkpoint=joinpath(output,"n03-samples-partial.csv")
            first_checkpoint=!isfile(checkpoint)
            open(checkpoint,"a") do io
                first_checkpoint && println(io,"case_id,feature,sample,time_ns,gc_time_ns,allocated_bytes,allocations,qualification")
                for i in eachindex(trial.times)
                    println(io,join((c.case_id,planned.feature.id,i,trial.times[i],trial.gctimes[i],trial.memory,trial.allocs,
                        "rational_oracle_passed_native_verdict_pending"),','))
                end
            end
            open(io->TOML.print(io,Dict("started_utc"=>started_utc,"source_sha256"=>fingerprint,
                "grid_sha256"=>n03_grid_sha256(),"selected_case_ids"=>getproperty.(cases,:case_id),
                "completed_case_ids"=>[row["case_id"] for row in rows],"status"=>"in_progress_not_final_qualification")),joinpath(output,"n03-progress.toml"),"w")
            PerfChecker.CheckerResult([PerfChecker.to_table(trial)],nothing,[:n03,:construction_included],
                [PerfChecker.PackageSpec(name="Garamon")],
                [Dict{String,Any}("correctness"=>Dict("status"=>"passed","required"=>true,
                    "message"=>"independent rational Chevalley/Gauss-Jordan inverse; all coefficients atol=1e-10 rtol=1e-9"),
                    "source_fingerprint"=>fingerprint,"execution"=>Dict("mode"=>"shared_process_solo", "construction_included"=>true))])
        end
        suite=SoftwareSuite(:n03_radical,[PackageSuite("Garamon";source=dirname(@__DIR__),
            worker_environment=joinpath(@__DIR__,"runner"),versions=VersionNumber[],dev_sources=String[],features)])
        result=run_suite(suite;profile=:quick,strict=false,executor)
        write_suite_json(result,joinpath(output,"n03-qualification.json"))
        open(joinpath(output,"n03-diagnostics.toml"),"w") do io;TOML.print(io,Dict("cases"=>rows));end
        open(joinpath(output,"n03-samples.csv"),"w") do io
            println(io,"feature,comparison_key,status,sample,time_ns,gc_time_ns,allocated_bytes,allocations")
            for run in result.runs
                run.result isa PerfChecker.CheckerResult || continue
                table=only(run.result.tables)
                for i in eachindex(table.times)
                    println(io,join((run.planned.feature.id,run.planned.comparison_key,run.status,i,
                        table.times[i],table.gctimes[i],table.memory[i],table.allocs[i]),','))
                end
            end
        end
        metadata=Dict("source_sha256"=>fingerprint,"source_unchanged"=>n03_fingerprint()==fingerprint,
            "started_utc"=>started_utc,"finished_utc"=>string(now(UTC))*"Z","grid_sha256"=>n03_grid_sha256(),"grid_version"=>N03_GRID_VERSION,
            "selected_case_ids"=>getproperty.(cases,:case_id),"completed_case_ids"=>[row["case_id"] for row in rows],
            "all_cases_passed"=>suite_passed(result),"selection_shard"=>(shard===nothing ? Int[] : collect(shard)),
            "selection_range"=>(interval===nothing ? Int[] : collect(interval)),
            "condition"=>condition.condition,"interference_label"=>condition.label,"isolation_certified"=>false,
            "julia"=>string(VERSION),"perfchecker"=>string(pkgversion(PerfChecker)),"cpu"=>Sys.cpu_info()[1].model,
            "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"controller_peak_rss_bytes"=>Sys.maxrss(),
            "elapsed_seconds"=>time()-started,"mode"=>string(mode),"reverse_order"=>reverse_order,
            "measured_cases"=>length(result.runs),"prepared_matrix_cases"=>length(matrix),
            "verdict"=>string(suite_verdict(result)),"kernel_independent_of_garamon_src"=>true)
        open(io->TOML.print(io,metadata),joinpath(output,"n03-environment.toml"),"w")
        rm(joinpath(output,"n03-samples-partial.csv");force=true) # complete native samples now archived
        sum(filesize(joinpath(d,f)) for (d,_,fs) in walkdir(output) for f in fs)<=64<<20 || error("N03 archive disk budget")
        metadata["source_unchanged"] || error("N03 source changed")
        suite_passed(result) || error("N03 has failed/refused cases; inspect qualification")
        println("N03 ",suite_verdict(result),": ",length(result.runs)," cases; ",round(time()-started;digits=3)," seconds")
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    modes=filter(arg->arg in ("--smoke","--screen","--full"),ARGS)
    length(modes)<=1 || error("choose one mode")
    mode=isempty(modes) ? :smoke : Symbol(first(modes)[3:end])
    cap=filter(arg->startswith(arg,"--max-cases="),ARGS)
    length(cap)<=1 || error("duplicate case budget")
    maximum_cases=isempty(cap) ? 800 : parse(Int,split(only(cap),'=';limit=2)[2])
    shards=filter(arg->startswith(arg,"--shard="),ARGS);ranges=filter(arg->startswith(arg,"--range="),ARGS)
    length(shards)<=1 && length(ranges)<=1 || error("duplicate selection")
    shard=isempty(shards) ? nothing : Tuple(parse.(Int,split(split(only(shards),'=';limit=2)[2],'/')))
    interval=isempty(ranges) ? nothing : Tuple(parse.(Int,split(split(only(ranges),'=';limit=2)[2],':')))
    shard===nothing || length(shard)==2 || error("shard must be INDEX/COUNT")
    interval===nothing || length(interval)==2 || error("range must be FIRST:LAST")
    known=Set(vcat(modes,cap,shards,ranges,["--prepare-only","--reverse"]))
    paths=filter(arg->!(arg in known),ARGS)
    length(paths)==1 && !startswith(only(paths),"--") || error("usage: n03_compare.jl [--smoke|--screen|--full] [--prepare-only] [--reverse] [--max-cases=N] OUTPUT")
    n03_campaign(abspath(only(paths));mode,prepare_only="--prepare-only" in ARGS,max_cases=maximum_cases,reverse_order="--reverse" in ARGS,shard,interval)
end
