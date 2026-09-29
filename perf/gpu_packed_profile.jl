# CUPTI diagnostics for the exact packed GPU pilot; timings are descriptive only.
include("gpu_packed_compare.jl")

function gpu_packed_profile(output;dimensions=(8,65),horizon=8192)
    VERSION.major==1 && VERSION.minor==13 || error("GPU profiling requires Julia 1.13")
    Threads.nthreads()==1 || error("GPU profiling requires one Julia thread")
    CUDA.functional() || error("no functional CUDA device")
    condition=get(ENV,"GARAMONBENCH_CONDITION","")
    label=strip(get(ENV,"GARAMONBENCH_INTERFERENCE_LABEL",""))
    condition in ("isolated","exploratory_interference") || error("declare measurement condition")
    condition=="isolated" ? isempty(label) || error("isolated label must be empty") :
        !isempty(label) || error("exploratory label required")
    output=abspath(output)
    ispath(output) && error("use a fresh output directory")
    mkpath(output)
    sources=[@__FILE__,joinpath(@__DIR__,"gpu_packed_compare.jl"),
        joinpath(@__DIR__,"gpu_packed.jl"),joinpath(@__DIR__,"binary_rank_compact_cases.jl"),
        joinpath(@__DIR__,"binary_rank_cases.jl")]
    sourcehash()=bytes2hex(sha256(join(read.(sources,String),"\n")))
    original=sourcehash()
    smi()=strip(read(`nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version,utilization.gpu --format=csv,noheader`,String))
    metadata=Dict{String,Any}("status"=>"running","condition"=>condition,
        "interference_label"=>label,"isolation_certified"=>false,
        "julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "device"=>string(CUDA.device()),"dimensions"=>collect(dimensions),
        "horizon"=>horizon,"family"=>"higher_rank","signature"=>"positive",
        "sources"=>sources,"source_sha256"=>original,"source_unchanged"=>false,
        "gpu_before"=>smi(),"free_device_memory_before_bytes"=>CUDA.free_memory(),
        "contract"=>"same packed path plan; CPU and GPU checked against independent Int64 word oracle; CUPTI trace is diagnostic, not a timing comparison",
        "profiles"=>["device_kernel_reused_output","resident_to_host_owned","complete_episode"],
        "threads"=>Threads.nthreads(),"argv"=>split(read("/proc/self/cmdline",String),'\0';keepempty=false))
    manifest=joinpath(output,"manifest.toml")
    open(io->TOML.print(io,metadata),manifest,"w")
    try
        for n in dimensions
            fixture=compact_fixture(n,:higher_rank,:positive)
            batch=gpu_make_batch(fixture,horizon)
            resident=gpu_resident_batch(batch;max_bytes=512<<20)
            gpu_exact_oracle(fixture,batch,run_packed_batch(batch),horizon) ||
                error("CPU oracle failed at $n dimensions")
            gpu_exact_oracle(fixture,batch,gpu_owned_matrix(resident),horizon) ||
                error("GPU oracle failed at $n dimensions")
            for (phase,operation) in (
                ("device_kernel_reused_output",()->gpu_run!(resident)),
                ("resident_to_host_owned",()->gpu_owned_matrix(resident)),
                ("complete_episode",()->gpu_complete_matrix(gpu_make_batch(fixture,horizon);max_bytes=512<<20)))
                file=joinpath(output,"n$(lpad(string(n),3,'0'))-$phase.txt")
                open(file,"w") do io
                    profile=CUDA.@profile trace=true operation()
                    show(io,MIME"text/plain"(),profile)
                end
            end
            gpu_exact_oracle(fixture,batch,gpu_owned_matrix(resident),horizon) ||
                error("GPU post-profile oracle failed at $n dimensions")
        end
        metadata["source_unchanged"]=sourcehash()==original
        metadata["status"]=metadata["source_unchanged"] ? "validated" : "failed"
        metadata["status"]=="validated" || error("source changed during GPU profile")
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
    length(ARGS)==1 || error("usage: gpu_packed_profile.jl NEW_OUTPUT_DIRECTORY")
    BLAS.set_num_threads(1)
    println(gpu_packed_profile(ARGS[1]))
end
