# CUPTI diagnosis of exact resident sums; instrumented times are not benchmarks.
include("resident_sum_compare.jl")

function resident_sum_profile(output;dimensions=(8,65),horizons=(32,1024),repetitions=32)
    VERSION.major==1 && VERSION.minor==13 || error("resident profiling requires Julia 1.13")
    Threads.nthreads()==4 || error("resident profiling requires four Julia threads")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated label must be empty") :
        !isempty(label) || error("exploratory label required")
    output=abspath(output);ispath(output) && error("use a fresh output directory")
    mkpath(output)
    sources=[@__FILE__,joinpath(@__DIR__,"resident_sum_compare.jl"),
        joinpath(@__DIR__,"resident_sum.jl"),joinpath(@__DIR__,"gpu_packed.jl"),
        joinpath(@__DIR__,"cpu_threaded_packed.jl"),
        joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank_cases.jl")]
    sourcehash()=bytes2hex(sha256(join(read.(sources,String),"\n")))
    original=sourcehash()
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running","condition"=>condition,
        "interference_label"=>label,"isolation_certified"=>false,
        "julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "device"=>string(CUDA.device()),"dimensions"=>collect(dimensions),
        "horizons"=>collect(horizons),"repetitions"=>repetitions,
        "family"=>"higher_rank","signature"=>"positive",
        "sources"=>sources,"source_sha256"=>original,"source_unchanged"=>false,
        "gpu_before"=>smi(),"free_device_memory_before_bytes"=>CUDA.free_memory(),
        "contract"=>"four resident packed input sets; same exact R products and one owned final host sum; CUDA CUPTI trace is diagnostic, not a speed ranking",
        "profiles"=>["R_kernel_gpu","fused_single_kernel_gpu","captured_graph_gpu"],
        "threads"=>Threads.nthreads(),"argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false))
    manifest=joinpath(output,"manifest.toml")
    open(io->TOML.print(io,metadata),manifest,"w")
    try
        for n in dimensions,horizon in horizons
            fixture=compact_fixture(n,:higher_rank,:positive)
            expected=sum_expected(fixture,horizon,repetitions)
            gpu=build_resident_sum_gpu(fixture,horizon)
            graph=build_resident_sum_graph(gpu,repetitions)
            resident_sum_gpu(gpu,repetitions)==expected || error("ordinary GPU oracle failed")
            resident_sum_gpu_fused(gpu,repetitions)==expected || error("fused GPU oracle failed")
            resident_sum_gpu_graph(graph)==expected || error("graph GPU oracle failed")
            for (phase,operation) in (
                ("R_kernel_gpu",()->resident_sum_gpu(gpu,repetitions)),
                ("fused_single_kernel_gpu",()->resident_sum_gpu_fused(gpu,repetitions)),
                ("captured_graph_gpu",()->resident_sum_gpu_graph(graph)))
                file=joinpath(output,"n$(lpad(string(n),3,'0'))-H$(horizon)-$phase.txt")
                open(file,"w") do io
                    profile=CUDA.@profile trace=true operation()
                    show(io,MIME"text/plain"(),profile)
                end
            end
            resident_sum_gpu(gpu,repetitions)==expected || error("ordinary post-profile oracle failed")
            resident_sum_gpu_fused(gpu,repetitions)==expected || error("fused post-profile oracle failed")
            resident_sum_gpu_graph(graph)==expected || error("graph post-profile oracle failed")
        end
        metadata["source_unchanged"]=sourcehash()==original
        metadata["status"]=metadata["source_unchanged"] ? "validated" : "failed"
        metadata["status"]=="validated" || error("source changed during profile")
    catch error
        metadata["status"]="failed"
        metadata["failure"]=sprint(showerror,error)
        rethrow()
    finally
        metadata["gpu_after"]=smi()
        metadata["free_device_memory_after_bytes"]=CUDA.free_memory()
        open(io->TOML.print(io,metadata),manifest,"w")
    end
    output
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: resident_sum_profile.jl NEW_OUTPUT_DIRECTORY")
    BLAS.set_num_threads(1)
    println(resident_sum_profile(ARGS[1]))
end
