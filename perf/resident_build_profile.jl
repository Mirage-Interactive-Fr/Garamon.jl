# Reproducible PerfChecker CPU and allocation flame graphs for resident build.
using Garamon, CUDA, PerfChecker, Profile, TOML, SHA, LinearAlgebra
include("resident_sum_compare.jl")

length(ARGS)==3 || error("usage: resident_build_profile.jl NEW_OUTPUT_DIRECTORY DIMENSION HORIZON")
const PROFILE_ROOT=abspath(ARGS[1])
const PROFILE_DIMENSION=parse(Int,ARGS[2])
const PROFILE_HORIZON=parse(Int,ARGS[3])
PROFILE_DIMENSION>=2 && PROFILE_HORIZON>=1 || error("positive profile dimensions required")
VERSION.major==1 && VERSION.minor==13 || error("profile requires Julia 1.13")
Threads.nthreads()==4 || error("profile requires four Julia threads")
ispath(PROFILE_ROOT) && error("profile output must be a fresh directory")
mkpath(PROFILE_ROOT)
BLAS.set_num_threads(1)
const PROFILE_FIXTURE=compact_fixture(PROFILE_DIMENSION,:higher_rank,:positive)
profile_build()=build_resident_sum_cpu(PROFILE_FIXTURE,PROFILE_HORIZON)

sourcefiles=sort(vcat([@__FILE__,joinpath(@__DIR__,"resident_sum_compare.jl"),
    joinpath(@__DIR__,"resident_sum.jl"),joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
    joinpath(@__DIR__,"binary_rank_cases.jl")],
    [joinpath(dirname(@__DIR__),"src",file)
     for file in readdir(joinpath(dirname(@__DIR__),"src")) if endswith(file,".jl")]))
fingerprint()=bytes2hex(sha256(join(read.(sourcefiles,String),"\n")))
original=fingerprint()
metadata=Dict{String,Any}(
    "status"=>"running","dimension"=>PROFILE_DIMENSION,"horizon"=>PROFILE_HORIZON,
    "family"=>"higher_rank","metric"=>"positive diagonal","julia"=>string(VERSION),
    "perfchecker"=>string(pkgversion(PerfChecker)),"threads"=>Threads.nthreads(),
    "blas_threads"=>BLAS.get_num_threads(),"source_files"=>sourcefiles,
    "source_sha256"=>original,"cpu_profile_seconds"=>2.0,"cpu_profile_delay"=>0.001,
    "allocation_sample_rate"=>0.1,"allocation_profile_repetitions"=>3,
    "source_unchanged"=>false,"rss_budget_bytes"=>6<<30)
manifest=joinpath(PROFILE_ROOT,"manifest.toml")
open(io->TOML.print(io,metadata),manifest,"w")

xml(text)=replace(string(text),"&"=>"&amp;","<"=>"&lt;",">"=>"&gt;","\""=>"&quot;")
mutable struct FlameNode
    label::String
    value::Float64
    children::Dict{String,FlameNode}
end
FlameNode(label)=FlameNode(label,0.0,Dict{String,FlameNode}())
function flame_svg(rows,valuefn,path,title)
    tree=FlameNode(title)
    depthmax=0
    for row in rows
        value=Float64(valuefn(row))
        value>0 || continue
        node=tree
        node.value+=value
        for (depth,label) in enumerate(first(row.stack,min(length(row.stack),18)))
            node=get!(node.children,label) do
                FlameNode(label)
            end
            node.value+=value
            depthmax=max(depthmax,depth)
        end
    end
    tree.value>0 || error("empty flame graph")
    width=1400.0
    rowheight=24
    height=(depthmax+2)*rowheight
    function paint(io,node,x,depth,w)
        y=height-(depth+1)*rowheight
        color=depth==0 ? "#30445c" :
            ("#f29e4c","#e87662","#db6687","#ac72b3","#7887c6")[mod1(depth,5)]
        println(io,"<rect x=\"",round(x;digits=2),"\" y=\"",y,
            "\" width=\"",round(max(w-0.5,0.1);digits=2),
            "\" height=\"22\" fill=\"",color,"\" stroke=\"#ffffff\" stroke-width=\"0.4\"/>")
        println(io,"<title>",xml(node.label)," — ",round(node.value;digits=1),"</title>")
        if w>85
            label=first(node.label,min(length(node.label),max(5,Int(floor((w-12)/7)))))
            println(io,"<text x=\"",round(x+4;digits=2),"\" y=\"",y+15,
                "\" font-size=\"11\" fill=\"white\">",xml(label),"</text>")
        end
        children=sort!(collect(values(node.children));by=child->(-child.value,child.label))
        cursor=x
        for child in children
            childwidth=w*child.value/node.value
            paint(io,child,cursor,depth+1,childwidth)
            cursor+=childwidth
        end
    end
    open(path,"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1400\" height=\"",
            height,"\" viewBox=\"0 0 1400 ",height,"\">")
        println(io,"<rect width=\"1400\" height=\"",height,"\" fill=\"#f7f9fc\"/>")
        paint(io,tree,0.0,0,width)
        println(io,"</svg>")
    end
end
function folded(rows,valuefn,path)
    open(path,"w") do io
        for row in rows
            value=valuefn(row)
            value>0 || continue
            println(io,join(row.stack,";")," ",round(Int,value))
        end
    end
end
function line_summary(rows,valuefn)
    groups=Dict{String,Float64}()
    for row in rows
        key=string(row.filename,":",row.line)
        groups[key]=get(groups,key,0.0)+Float64(valuefn(row))
    end
    [Dict("site"=>site,"value"=>value) for (site,value) in
        first(sort!(collect(groups);by=x->(-last(x),first(x))),min(20,length(groups)))]
end

try
    first_call=@timed profile_build()
    metadata["first_call_time_ns"]=first_call.time*1e9
    metadata["first_call_compile_ns"]=first_call.compile_time*1e9
    resident_sum_cpu(first_call.value,8;threaded=true)==
        sum_expected(PROFILE_FIXTURE,PROFILE_HORIZON,8) ||
        error("pre-profile integer oracle failed")
    cpu_options=merge(PerfChecker.default_options(Val(:profile)),
        Dict(:targets=>["Garamon"],:profile_seconds=>2.0,
            :profile_delay=>0.001,:max_profile_stacks=>2000,:repeat=>true))
    cpu=Core.eval(Main,PerfChecker.check(cpu_options,:(profile_build()),Val(:profile)))
    alloc_options=merge(PerfChecker.default_options(Val(:profile_alloc)),
        Dict(:targets=>["Garamon"],:sample_rate=>0.1,
            :profile_repetitions=>3,:max_profile_stacks=>2000,:repeat=>true))
    alloc=Core.eval(Main,PerfChecker.check(alloc_options,:(profile_build()),Val(:profile_alloc)))
    isempty(cpu) && error("empty CPU profile")
    isempty(alloc) && error("empty allocation profile")
    folded(cpu,row->row.samples,joinpath(PROFILE_ROOT,"cpu.folded"))
    folded(alloc,row->row.bytes,joinpath(PROFILE_ROOT,"alloc.folded"))
    flame_svg(cpu,row->row.samples,joinpath(PROFILE_ROOT,"cpu-flamegraph.svg"),
        "PerfChecker CPU samples")
    flame_svg(alloc,row->row.bytes,joinpath(PROFILE_ROOT,"allocation-flamegraph.svg"),
        "PerfChecker sampled allocation bytes")
    metadata["cpu_profile_rows"]=length(cpu)
    metadata["allocation_profile_rows"]=length(alloc)
    metadata["cpu_profile_total_samples"]=sum(row->row.samples,cpu)
    metadata["allocation_profile_estimated_bytes"]=sum(row->row.bytes,alloc)
    metadata["cpu_top_lines"]=line_summary(cpu,row->row.samples)
    metadata["allocation_top_lines"]=line_summary(alloc,row->row.bytes)
    resident_sum_cpu(profile_build(),8;threaded=true)==
        sum_expected(PROFILE_FIXTURE,PROFILE_HORIZON,8) ||
        error("post-profile integer oracle failed")
    metadata["source_unchanged"]=fingerprint()==original
    metadata["rss_bytes"]=Sys.maxrss()
    metadata["status"]=metadata["source_unchanged"] &&
        metadata["rss_bytes"]<=metadata["rss_budget_bytes"] ? "validated" : "failed"
    metadata["status"]=="validated" || error("profile qualification failed")
finally
    metadata["rss_bytes"]=Sys.maxrss()
    open(io->TOML.print(io,metadata),manifest,"w")
end
println(PROFILE_ROOT)
