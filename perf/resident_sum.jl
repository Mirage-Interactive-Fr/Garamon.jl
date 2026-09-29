"""Exact long-session sum of packed products with four resident operand batches."""
module ResidentSumPrototype
using Garamon, CUDA, LinearAlgebra
include("cpu_threaded_packed.jl")
include("gpu_packed.jl")
using .CPUThreadedPackedPrototype, .GPUPackedPrototype

export build_resident_sum_cpu, build_resident_sum_gpu, resident_sum_cpu,
    resident_sum_gpu, resident_sum_gpu_fused, resident_sum_packs,
    GPUResidentSumGraph, build_resident_sum_graph, resident_sum_gpu_graph

function resident_sum_packs(fixture,horizon)
    horizon>=1 || throw(ArgumentError("resident sum requires at least one column"))
    a,b=first(fixture.inputs)
    plan=prepare_product(a,b)
    [begin
        left=[fixture.inputs[mod1(column+shift,4)][1] for column in 1:horizon]
        right=[fixture.inputs[mod1(column+shift,4)][2] for column in 1:horizon]
        pack_product_batch(plan,left,right)
     end for shift in 0:3]
end

function build_resident_sum_cpu(fixture,horizon;max_bytes::Integer=512<<20)
    batches=resident_sum_packs(fixture,horizon)
    residents=map(batch->cpu_resident_batch(batch;max_bytes),batches)
    bytes=sum(big(sizeof(r.source.left_values))+sizeof(r.source.right_values)+
        sizeof(r.output) for r in residents)
    bytes<=max_bytes || throw(ArgumentError("combined CPU residents exceed budget"))
    residents
end

function build_resident_sum_gpu(fixture,horizon;max_bytes::Integer=512<<20)
    batches=resident_sum_packs(fixture,horizon)
    residents=map(batch->gpu_resident_batch(batch;max_bytes),batches)
    bytes=sum(big(sizeof(r.left))+sizeof(r.right)+sizeof(r.output)+
        sizeof(r.paths.starts)+sizeof(r.paths.ai)+sizeof(r.paths.bi)+
        sizeof(r.paths.factor) for r in residents)
    bytes+=sizeof(first(residents).output)
    bytes<=max_bytes || throw(ArgumentError("combined GPU residents exceed budget"))
    residents
end

"""Sum every contribution of every product; return a newly owned host matrix."""
function resident_sum_cpu(residents,repetitions;threaded::Bool=false)
    length(residents)==4 && repetitions>=1 || throw(ArgumentError("four residents and positive repetitions required"))
    firstbatch=first(residents).source
    O=length(firstbatch.plan.output_masks);H=size(firstbatch.left_values,2)
    output=zeros(eltype(firstbatch.left_values),O,H)
    for step in 1:repetitions
        batch=residents[mod1(step,4)].source
        if threaded
            Threads.@threads :static for column in 1:H
                for (ai,bi,oi,factor) in batch.plan.paths
                    @inbounds output[oi,column]+=factor*batch.left_values[ai,column]*
                        batch.right_values[bi,column]
                end
            end
        else
            for column in 1:H
                for (ai,bi,oi,factor) in batch.plan.paths
                    @inbounds output[oi,column]+=factor*batch.left_values[ai,column]*
                        batch.right_values[bi,column]
                end
            end
        end
    end
    output
end

function _accumulate_kernel!(output,left,right,starts,ai,bi,factor)
    linear=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    linear>length(output) && return nothing
    O=Int32(size(output,1))
    oi=(linear-Int32(1))%O+Int32(1)
    column=(linear-Int32(1))÷O+Int32(1)
    @inbounds value=output[oi,column]
    @inbounds for path in starts[oi]:(starts[oi+Int32(1)]-Int32(1))
        value+=factor[path]*left[ai[path],column]*right[bi[path],column]
    end
    @inbounds output[oi,column]=value
    return nothing
end

"""Keep operands and partial sum on device; copy only the final owned output."""
function resident_sum_gpu(residents,repetitions)
    length(residents)==4 && repetitions>=1 || throw(ArgumentError("four residents and positive repetitions required"))
    O,H=size(first(residents).output)
    output=CUDA.zeros(Float64,O,H)
    cells=length(output)
    for step in 1:repetitions
        resident=residents[mod1(step,4)]
        p=resident.paths
        @cuda threads=256 blocks=cld(cells,256) _accumulate_kernel!(output,resident.left,
            resident.right,p.starts,p.ai,p.bi,p.factor)
    end
    CUDA.synchronize()
    Array(output)
end

@inline function _accumulate_paths(value,left,right,starts,ai,bi,factor,oi,column)
    @inbounds for path in starts[oi]:(starts[oi+Int32(1)]-Int32(1))
        value+=factor[path]*left[ai[path],column]*right[bi[path],column]
    end
    value
end

function _fused_sum_kernel!(output,l1,r1,l2,r2,l3,r3,l4,r4,
    starts,ai,bi,factor,repetitions::Int32)
    linear=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    linear>length(output) && return nothing
    O=Int32(size(output,1))
    oi=(linear-Int32(1))%O+Int32(1)
    column=(linear-Int32(1))÷O+Int32(1)
    value=zero(eltype(output))
    for step in Int32(1):repetitions
        slot=(step-Int32(1))&Int32(3)
        if slot==Int32(0)
            value=_accumulate_paths(value,l1,r1,starts,ai,bi,factor,oi,column)
        elseif slot==Int32(1)
            value=_accumulate_paths(value,l2,r2,starts,ai,bi,factor,oi,column)
        elseif slot==Int32(2)
            value=_accumulate_paths(value,l3,r3,starts,ai,bi,factor,oi,column)
        else
            value=_accumulate_paths(value,l4,r4,starts,ai,bi,factor,oi,column)
        end
    end
    @inbounds output[oi,column]=value
    return nothing
end

"""Fuse all exact resident products into one GPU launch and one owned host result."""
function resident_sum_gpu_fused(residents,repetitions)
    length(residents)==4 && 1<=repetitions<=typemax(Int32) ||
        throw(ArgumentError("four residents and positive Int32 repetitions required"))
    plan=first(residents).source.plan
    dims=size(first(residents).output)
    all(r->r.source.plan===plan && size(r.output)==dims,residents) ||
        throw(ArgumentError("resident plans and output shapes must agree"))
    output=CuArray{Float64}(undef,dims)
    a,b,c,d=residents
    p=a.paths
    @cuda threads=256 blocks=cld(length(output),256) _fused_sum_kernel!(output,
        a.left,a.right,b.left,b.right,c.left,c.right,d.left,d.right,
        p.starts,p.ai,p.bi,p.factor,Int32(repetitions))
    CUDA.synchronize()
    Array(output)
end

struct GPUResidentSumGraph{R,O,G}
    residents::R
    output::O
    executable::G
    repetitions::Int32
end

"""Prepare a reusable exact CUDA graph; count capture and instantiation in build time."""
function build_resident_sum_graph(residents,repetitions;max_nodes::Integer=1024)
    length(residents)==4 && 1<=repetitions<=typemax(Int32) &&
        repetitions+1<=max_nodes ||
        throw(ArgumentError("four residents and positive bounded graph size required"))
    plan=first(residents).source.plan
    dims=size(first(residents).output)
    all(r->r.source.plan===plan && size(r.output)==dims,residents) ||
        throw(ArgumentError("resident plans and output shapes must agree"))
    output=CUDA.zeros(Float64,dims)
    cells=length(output)
    firstpaths=first(residents).paths
    fill!(output,0.0)
    @cuda threads=256 blocks=cld(cells,256) _accumulate_kernel!(output,
        first(residents).left,first(residents).right,firstpaths.starts,
        firstpaths.ai,firstpaths.bi,firstpaths.factor)
    CUDA.synchronize()
    graph=CUDA.capture() do
        fill!(output,0.0)
        for step in 1:repetitions
            resident=residents[mod1(step,4)]
            p=resident.paths
            @cuda threads=256 blocks=cld(cells,256) _accumulate_kernel!(output,
                resident.left,resident.right,p.starts,p.ai,p.bi,p.factor)
        end
    end
    GPUResidentSumGraph(residents,output,CUDA.instantiate(graph),Int32(repetitions))
end

"""Replay an already instantiated exact graph and return a new owned host matrix."""
function resident_sum_gpu_graph(graph::GPUResidentSumGraph)
    CUDA.launch(graph.executable)
    CUDA.synchronize()
    Array(graph.output)
end

end
