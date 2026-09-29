# Reproducible analysis of two fresh, oppositely ordered N03 screen archives.
# No benchmark is launched. Ratios are descriptive observations under recorded conditions.
using PerfChecker, Statistics, TOML
include("n03_shards.jl")
function n03_analyze(forward,reverse,output)
    ispath(output) && !isempty(readdir(output)) && error("fresh analysis output required")
    archives=[forward,reverse];metadata=[TOML.parsefile(joinpath(p,"n03-environment.toml")) for p in archives]
    @assert metadata[1]["source_sha256"]==metadata[2]["source_sha256"]
    @assert metadata[1]["condition"]==metadata[2]["condition"]
    @assert metadata[1]["interference_label"]==metadata[2]["interference_label"]
    @assert !metadata[1]["reverse_order"] && metadata[2]["reverse_order"]
    @assert all(m->m["threads"]==1 && m["blas_threads"]==1,metadata)
    audits=[n03_audit(p;mode=:screen) for p in archives]
    @assert all(a->a["complete"],audits)
    rows=[TOML.parsefile(joinpath(p,"n03-diagnostics.toml"))["cases"] for p in archives]
    for (path,data) in zip(archives,rows)
        native=PerfChecker.JSON.parse(read(joinpath(path,"n03-qualification.json"),String))
        @assert native["passed"] && length(native["runs"])==756
        @assert all(r->r["status"]=="pass" && r["qualification"]["correctness"]["status"]=="passed",native["runs"])
        @assert Set(r["feature"] for r in native["runs"])==Set(r["feature"] for r in data)
        lines=readlines(joinpath(path,"n03-samples.csv"));@assert length(lines)==1+756*7
        samplecounts=Dict{String,Int}()
        for line in lines[2:end]
            fields=split(line,',');@assert fields[3]=="pass"
            samplecounts[fields[1]]=get(samplecounts,fields[1],0)+1
        end
        @assert length(samplecounts)==756 && all(==(7),values(samplecounts))
    end
    byid=[Dict(r["case_id"]=>r for r in data) for data in rows]
    pairs=Dict{String,Any}[]
    for c in n03_cases(:screen)
        c.strategy==:regular || continue
        regular=[d[c.case_id] for d in byid];radical=[d[c.case_id+1] for d in byid]
        ratios=[regular[i]["median_ns"]/radical[i]["median_ns"] for i in 1:2]
        push!(pairs,Dict("regular_case_id"=>c.case_id,"dimension"=>c.n,"radical_dimension"=>c.r,
            "active_nonradical"=>c.s,"family"=>string(c.family),"horizon"=>c.horizon,
            "forward_regular_ns"=>regular[1]["median_ns"],"forward_radical_ns"=>radical[1]["median_ns"],
            "reverse_regular_ns"=>regular[2]["median_ns"],"reverse_radical_ns"=>radical[2]["median_ns"],
            "forward_regular_over_radical"=>ratios[1],"reverse_regular_over_radical"=>ratios[2],
            "ratio_change_between_orders"=>max(ratios...)/min(ratios...),
            "forward_regular_bytes"=>regular[1]["allocated_bytes"],"forward_radical_bytes"=>radical[1]["allocated_bytes"],
            "reverse_regular_bytes"=>regular[2]["allocated_bytes"],"reverse_radical_bytes"=>radical[2]["allocated_bytes"],
            "radical_faster_both_orders"=>all(>(1),ratios),"regular_faster_both_orders"=>all(<(1),ratios)))
    end
    mkpath(output)
    columns=["regular_case_id","dimension","radical_dimension","active_nonradical","family","horizon",
        "forward_regular_ns","forward_radical_ns","reverse_regular_ns","reverse_radical_ns",
        "forward_regular_over_radical","reverse_regular_over_radical","ratio_change_between_orders",
        "forward_regular_bytes","forward_radical_bytes","reverse_regular_bytes","reverse_radical_bytes",
        "radical_faster_both_orders","regular_faster_both_orders"]
    open(joinpath(output,"n03-paired-observations.csv"),"w") do io
        println(io,join(columns,','));for p in pairs;println(io,join((p[k] for k in columns),','));end
    end
    groups=Dict{String,Any}[]
    for r in 1:3,s in 1:2
        subset=filter(p->p["radical_dimension"]==r && p["active_nonradical"]==s,pairs)
        ratios=vcat([Float64[p["forward_regular_over_radical"],p["reverse_regular_over_radical"]] for p in subset]...)
        push!(groups,Dict("r"=>r,"s"=>s,"paired_conditions"=>length(subset),
            "minimum_ratio"=>minimum(ratios),"median_ratio_descriptive"=>median(ratios),"maximum_ratio"=>maximum(ratios),
            "radical_faster_both_orders"=>count(p->p["radical_faster_both_orders"],subset),
            "regular_faster_both_orders"=>count(p->p["regular_faster_both_orders"],subset)))
    end
    summary=Dict("status"=>"exploratory_two_orders_no_stable_threshold_or_speed_claim",
        "analysis_source_sha256"=>bytes2hex(sha256(read(@__FILE__))),"archives"=>basename.(archives),
        "source_sha256"=>metadata[1]["source_sha256"],"grid_sha256"=>n03_grid_sha256(),
        "condition"=>metadata[1]["condition"],"interference_label"=>metadata[1]["interference_label"],
        "cases_per_order"=>756,"samples_per_order"=>5292,"paired_conditions"=>length(pairs),
        "elapsed_seconds_by_order"=>[m["elapsed_seconds"] for m in metadata],
        "peak_rss_bytes_by_order"=>[m["controller_peak_rss_bytes"] for m in metadata],
        "maximum_absolute_error"=>maximum(r["maximum_coefficient_absolute_error"] for data in rows for r in data),
        "maximum_first_episode_allocated_bytes"=>maximum(r["first_episode_allocated_bytes"] for data in rows for r in data),
        "maximum_fixture_retained_bytes"=>maximum(r["fixture_retained_bytes"] for data in rows for r in data),
        "maximum_p95_over_median"=>maximum(r["p95_ns"]/r["median_ns"] for data in rows for r in data),
        "maximum_order_ratio_change"=>maximum(p["ratio_change_between_orders"] for p in pairs),
        "radical_faster_both_orders"=>count(p->p["radical_faster_both_orders"],pairs),
        "regular_faster_both_orders"=>count(p->p["regular_faster_both_orders"],pairs),
        "route_order_sensitive_or_tied"=>count(p->!p["radical_faster_both_orders"]&&!p["regular_faster_both_orders"],pairs),
        "native_qualification_and_raw_samples_verified"=>true,"groups"=>groups)
    for (i,audit) in enumerate(audits)
        audit["native_qualification_and_raw_samples_verified"]=true
        open(io->TOML.print(io,audit),joinpath(output,i==1 ? "n03-audit-forward.toml" : "n03-audit-reverse.toml"),"w")
    end
    open(io->TOML.print(io,summary),joinpath(output,"n03-screen-summary.toml"),"w")
    cp(@__FILE__,joinpath(output,"n03_analyze.jl"))
    TOML.print(stdout,summary)
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: n03_analyze.jl FORWARD REVERSE NEW_OUTPUT")
    n03_analyze(abspath.(ARGS)...)
end
