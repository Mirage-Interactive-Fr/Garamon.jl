using SHA, TOML
const N03_GRID_VERSION = "N03-grid-v2-signed-one-core"
const N03_SCREEN_DIMS = (2,3,4,5,8,12,16,32,64,65,96,128)
const N03_SHARD_CAP = 32
function n03_grid()
    tuples=[(;n,r,s,family,horizon,strategy) for n in 2:128 for r in 1:3 for s in 1:3
        for family in (:positive,:signed) for horizon in (1,8,32,128)
        for strategy in (:regular,:radical) if n>=r+s]
    [(;case_id=i,c...) for (i,c) in enumerate(tuples)]
end
function n03_grid_sha256(grid=n03_grid())
    bytes2hex(sha256(N03_GRID_VERSION*"\n"*join((join(values(c),',') for c in grid),'\n')))
end
function n03_cases(mode)
    grid=n03_grid()
    mode==:full && return grid
    mode==:screen && return filter(c->c.n in N03_SCREEN_DIMS && c.s<=2 && c.horizon in (1,8,32),grid)
    mode==:smoke && return filter(c->c.n==c.r+2 && c.s==2 && c.family==:signed && c.horizon==1,grid)
    error("unknown N03 profile")
end
function n03_select(mode;shard=nothing,interval=nothing,reverse_order=false,prepare_only=false)
    shard!==nothing && interval!==nothing && error("shard and range are exclusive")
    selected=shard!==nothing || interval!==nothing
    selected && mode!=:full && error("shard/range requires full profile")
    mode==:full && !selected && !prepare_only && error("full requires bounded shard/range; use prepare-only for its manifest")
    cases=n03_cases(mode)
    if shard!==nothing
        index,count=shard
        index isa Integer && count isa Integer && !(index isa Bool) && !(count isa Bool) || error("integer shard required")
        1<=index<=count<=length(cases) || error("invalid shard bounds")
        cases=cases[index:count:end]
    elseif interval!==nothing
        first,last=interval
        first isa Integer && last isa Integer && !(first isa Bool) && !(last isa Bool) || error("integer range required")
        1<=first<=last<=length(cases) || error("invalid range bounds")
        cases=cases[first:last]
    end
    selected && length(cases)>N03_SHARD_CAP && error("N03 shard exceeds 32 cases")
    reverse_order && reverse!(cases)
    cases
end
function n03_write_cases(path,cases)
    open(path,"w") do io
        println(io,"case_id,dimension,radical_dimension,active_nonradical,family,horizon,strategy,construction_included")
        for c in cases;println(io,join((values(c)...,true),','));end
    end
end
"""Structural audit of fresh native run archives for ONE order/condition/source repetition.
Qualification is read from each run's declared native verdict; this does not recompute it.
"""
function n03_audit(root;mode=:full)
    expected=getproperty.(n03_cases(mode),:case_id);ids=Int[];contexts=Set();problems=String[];runs=0
    for (dir,_,files) in walkdir(root)
        "n03-environment.toml" in files || continue
        m=TOML.parsefile(joinpath(dir,"n03-environment.toml"));runs+=1
        selected=get(m,"selected_case_ids",Int[]);completed=get(m,"completed_case_ids",Int[])
        append!(ids,completed)
        selected==completed || push!(problems,"incomplete: $dir")
        get(m,"source_unchanged",false) || push!(problems,"changed source: $dir")
        get(m,"all_cases_passed",false) || push!(problems,"unqualified: $dir")
        get(m,"grid_sha256","")==n03_grid_sha256() || push!(problems,"grid mismatch: $dir")
        push!(contexts,(get(m,"source_sha256",""),get(m,"condition",""),get(m,"interference_label",""),get(m,"reverse_order",false)))
        for name in ("n03-qualification.json","n03-samples.csv","n03-selection.csv")
            isfile(joinpath(dir,name)) || push!(problems,"missing $name: $dir")
        end
    end
    missing=sort!(collect(setdiff(Set(expected),Set(ids))));extra=sort!(collect(setdiff(Set(ids),Set(expected))))
    duplicate=sort!([id for id in unique(ids) if count(==(id),ids)>1])
    length(contexts)==1 || push!(problems,"mixed or absent source/condition/order contexts")
    Dict("mode"=>string(mode),"expected_cases"=>length(expected),"completed_records"=>length(ids),"runs"=>runs,
        "missing_ids"=>missing,"extra_ids"=>extra,"duplicate_ids"=>duplicate,"problems"=>problems,
        "complete"=>isempty(missing)&&isempty(extra)&&isempty(duplicate)&&isempty(problems),
        "qualification_scope"=>"structural coverage and recorded native verdict; no recomputation")
end
