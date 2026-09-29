# Replay audit; keeps each passage/order separate. No benchmark is launched.
using PerfChecker,TOML,Statistics
include("n05_shared_screen.jl")

function n05_screen_audit(roots;output=nothing,prefer_later_passage=false)
    grid=n05_screen_grid();grid_hash=n05_screen_grid_hash(grid)
    selected=Set(r.case_id for r in grid if r.strategy in (:independent,:shared_workspace))
    observations=NamedTuple[];problems=String[];contexts=Dict{String,Any}[];seen=Dict{Int,Int}()
    for root in roots
        sources=Set{String}();conditions=Set{Tuple{String,String}}()
        for directory in sort(readdir(root;join=true))
            isdir(directory) && startswith(basename(directory),"n05-shared-screen-batch-") || continue
            envfile=joinpath(directory,"n05-shared-screen-environment.toml")
            isfile(envfile) || (push!(problems,"missing metadata: $directory");continue)
            env=TOML.parsefile(envfile)
            if !isempty(env["completed_case_ids"])
                version=VersionNumber(env["julia"])
                version.major==1 && version.minor==13 || error("official K3 audit requires Julia 1.13")
            end
            env["grid_sha256"]==grid_hash || error("grid mismatch in $directory")
            push!(sources,env["source_sha256"])
            push!(conditions,(env["measurement_condition"],env["interference_label"]))
            samples=joinpath(directory,"n05-shared-screen-samples.csv")
            rows=isfile(samples) ? split.(readlines(samples)[2:end],',') : Vector{Vector{SubString{String}}}()
            for id in env["completed_case_ids"]
                id in selected || error("unexpected route in exploratory audit")
                r=grid[id];case_rows=filter(x->parse(Int,x[1])==id,rows)
                length(case_rows)==15 && all(x->length(x)==17 && x[12]=="pass",case_rows) ||
                    (push!(problems,"incomplete/failed samples: $directory/$id");continue)
                sort(parse.(Int,getindex.(case_rows,13)))==collect(1:15) || error("sample numbering $directory/$id")
                all(x->join(x[1:11],',')==join(values(r),','),case_rows) || error("fixture mismatch $directory/$id")
                qualification=joinpath(directory,"case-$id-qualification.json")
                trace=joinpath(directory,"case-$id-diagnostics.toml")
                isfile(qualification)&&isfile(trace) || (push!(problems,"missing evidence: $directory/$id");continue)
                # Pinned PerfChecker JSON adapter, used only for reading its own evidence.
                q=PerfChecker._json_parsefile(qualification);run=only(q["runs"]);details=TOML.parsefile(trace)
                q["passed"] && run["status"]=="pass" && run["qualification"]["correctness"]["status"]=="passed" && details["qualified"] ||
                    (push!(problems,"oracle not qualified: $directory/$id");continue)
                ns=parse.(Float64,getindex.(case_rows,14));bytes=parse.(Int,getindex.(case_rows,16));allocations=parse.(Int,getindex.(case_rows,17))
                all(isfinite,ns)&&all(>=(0),ns) || error("invalid timing")
                push!(observations,(;passage=basename(root),r...,median_ns=median(ns),median_bytes=median(bytes),
                    median_allocations=median(allocations),builds=details["contraction_builds"],pairs=details["pair_slots"],
                    first_ms=details["first_episode_ms"],compile_ms=details["compile_ms"],
                    peak_rss_bytes=details["worker_peak_rss_bytes"],retained_bytes=details["retained_bytes"],
                    prepare_ms=haskey(details,"warm_preparation_constructor_probe_ms") ? median(details["warm_preparation_constructor_probe_ms"]) : NaN,
                    max_error=parse(BigFloat,details["max_absolute_error"])))
                seen[id]=get(seen,id,0)+1
            end
        end
        length(sources)<=1 && length(conditions)<=1 || error("mixed context within passage $root")
        push!(contexts,Dict("passage"=>basename(root),"sources"=>collect(sources),"conditions"=>[collect(c) for c in conditions]))
    end
    covered=Set(keys(seen));missing=sort!(collect(setdiff(selected,covered)))
    duplicates=sort!([id for (id,count) in seen if count>1])
    raw_observations=observations
    last_index=Dict(r.case_id=>i for (i,r) in enumerate(raw_observations))
    excluded=NamedTuple[]
    if prefer_later_passage
        excluded=[r for (i,r) in enumerate(raw_observations) if last_index[r.case_id]!=i]
        observations=[r for (i,r) in enumerate(raw_observations) if last_index[r.case_id]==i]
    end
    paired=NamedTuple[]
    keys_to_pair=unique((r.passage,r.n,r.k,r.horizon,r.changes,r.order) for r in observations)
    for key in keys_to_pair
        matches=filter(r->(r.passage,r.n,r.k,r.horizon,r.changes,r.order)==key,observations)
        length(matches)==2 && Set(r.strategy for r in matches)==Set((:independent,:shared_workspace)) || continue
        baseline=only(filter(r->r.strategy==:independent,matches))
        workspace=only(filter(r->r.strategy==:shared_workspace,matches))
        push!(paired,(;passage=key[1],n=key[2],k=key[3],horizon=key[4],changes=key[5],order=key[6],
            independent_ns=baseline.median_ns,workspace_ns=workspace.median_ns,
            workspace_over_independent=workspace.median_ns/baseline.median_ns,
            independent_bytes=baseline.median_bytes,workspace_bytes=workspace.median_bytes,
            independent_builds=baseline.builds,workspace_builds=workspace.builds,
            workspace_preparation_ms=workspace.prepare_ms))
    end
    summary=Dict{String,Any}("grid_sha256"=>grid_hash,"expected_cases"=>length(selected),
        "qualified_unique_cases"=>length(covered),"raw_samples"=>15length(raw_observations),
        "selected_samples"=>15length(observations),"selected_case_observations"=>length(observations),
        "duplicate_policy"=>prefer_later_passage ? "later passage selected by supplied root order; raw observations retained" : "duplicates refuse complete status",
        "excluded_observations"=>[Dict("case_id"=>r.case_id,"passage"=>r.passage) for r in excluded],
        "missing_case_ids"=>missing,"duplicate_case_ids"=>duplicates,"problems"=>problems,"passages"=>contexts,
        "covered_dimensions"=>sort!(unique([grid[id].n for id in covered])),
        "paired_comparisons_within_passage"=>length(paired),"unpaired_timing_observations"=>length(observations)-2length(paired),
        "complete"=>isempty(missing)&&(isempty(duplicates)||prefer_later_passage)&&isempty(problems),
        "timings_pooled_between_passages"=>false)
    if output!==nothing
        mkpath(output)
        open(io->TOML.print(io,summary),joinpath(output,"n05-shared-screen-audit.toml"),"w")
        open(joinpath(output,"n05-shared-screen-case-summary.csv"),"w") do io
            isempty(raw_observations) || println(io,join(keys(first(raw_observations)),','))
            for row in raw_observations;println(io,join(values(row),','));end
        end
        open(joinpath(output,"n05-shared-screen-selected-case-summary.csv"),"w") do io
            isempty(observations) || println(io,join(keys(first(observations)),','))
            for row in observations;println(io,join(values(row),','));end
        end
        open(joinpath(output,"n05-shared-screen-paired-summary.csv"),"w") do io
            isempty(paired) || println(io,join(keys(first(paired)),','))
            for row in paired;println(io,join(values(row),','));end
        end
    end
    (;summary,observations,raw_observations,paired)
end

if abspath(PROGRAM_FILE)==@__FILE__
    args=filter(!=("--prefer-later-passage"),ARGS)
    length(args)>=2 || error("usage: audit [--prefer-later-passage] output_directory passage_root...")
    result=n05_screen_audit(args[2:end];output=args[1],prefer_later_passage="--prefer-later-passage" in ARGS)
    println("K3 audit: ",result.summary["qualified_unique_cases"],"/",result.summary["expected_cases"],
        "; missing ",length(result.summary["missing_case_ids"]),"; problems ",length(result.summary["problems"]))
end
