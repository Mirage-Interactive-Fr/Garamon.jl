"""Experimental exact packed-product route for independent batch columns on CUDA."""
module GPUPackedPrototype
using CUDA, Garamon, LinearAlgebra

export GPUResidentBatch, gpu_resident_batch, gpu_run!, gpu_owned_matrix,
    gpu_enqueue!, gpu_owned_matrix_pinned, gpu_complete_matrix, gpu_complete_matrix_pinned

struct GPUResidentBatch{B,L,R,O,P}
    source::B
    left::L
    right::R
    output::O
    paths::P
end

"""Group paths by output while preserving their original order within each output."""
function _output_buckets(plan::ProductPlan)
    A=length(plan.left_masks);B=length(plan.right_masks);O=length(plan.output_masks)
    O>=1 || throw(ArgumentError("GPU pilot requires at least one reachable output"))
    O<=typemax(Int32) && A<=typemax(Int32) && B<=typemax(Int32) &&
        length(plan.paths)<=typemax(Int32)-1 || throw(ArgumentError("GPU index range exceeds Int32"))
    groups=[Tuple{Int32,Int32,Float64}[] for _ in 1:O]
    for (ai,bi,oi,factor) in plan.paths
        1<=ai<=A && 1<=bi<=B && 1<=oi<=O && isfinite(factor) ||
            throw(ArgumentError("GPU path index or factor invalid"))
        push!(groups[oi],(Int32(ai),Int32(bi),Float64(factor)))
    end
    starts=Vector{Int32}(undef,O+1)
    ai=Int32[];bi=Int32[];factor=Float64[]
    for oi in 1:O
        starts[oi]=Int32(length(ai)+1)
        for (a,b,f) in groups[oi]
            push!(ai,a);push!(bi,b);push!(factor,f)
        end
    end
    starts[end]=Int32(length(ai)+1)
    (;starts,ai,bi,factor)
end

function _packed_kernel!(output,left,right,starts,ai,bi,factor)
    linear=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    linear>length(output) && return nothing
    O=Int32(size(output,1))
    oi=(linear-Int32(1))%O+Int32(1)
    column=(linear-Int32(1))÷O+Int32(1)
    value=zero(eltype(output))
    @inbounds for path in starts[oi]:(starts[oi+Int32(1)]-Int32(1))
        value+=factor[path]*left[ai[path],column]*right[bi[path],column]
    end
    @inbounds output[oi,column]=value
    return nothing
end

"""Move a validated Float64 packed batch and stable path ordering to the GPU."""
function gpu_resident_batch(batch::PackedProductBatch;max_bytes::Integer=1<<30)
    CUDA.functional() || throw(ArgumentError("CUDA is unavailable on this host"))
    max_bytes>0 || throw(ArgumentError("GPU memory budget must be positive"))
    plan=batch.plan
    eltype(batch.left_values)==Float64 && eltype(batch.right_values)==Float64 &&
        eltype(plan.diagonal)==Float64 || throw(ArgumentError("GPU pilot accepts Float64"))
    size(batch.left_values,1)==length(plan.left_masks) &&
        size(batch.right_values,1)==length(plan.right_masks) &&
        size(batch.left_values,2)==size(batch.right_values,2) ||
        throw(DimensionMismatch("GPU packed input shape differs from plan"))
    isdiag(Garamon.metric(plan.algebra)) &&
        diag(Garamon.metric(plan.algebra))==plan.diagonal &&
        Garamon.basis(plan.algebra)==plan.basis_names ||
        throw(ArgumentError("GPU packed product plan algebra changed"))
    buckets=_output_buckets(plan)
    O=length(plan.output_masks);H=size(batch.left_values,2)
    cells=big(O)*H
    cells<=typemax(Int32) || throw(ArgumentError("GPU launch range exceeds Int32"))
    bytes=big(sizeof(batch.left_values))+sizeof(batch.right_values)+8cells+
        sizeof(buckets.starts)+sizeof(buckets.ai)+sizeof(buckets.bi)+sizeof(buckets.factor)
    bytes<=max_bytes || throw(ArgumentError("GPU workspace exceeds declared byte budget"))
    paths=(;starts=CuArray(buckets.starts),ai=CuArray(buckets.ai),
        bi=CuArray(buckets.bi),factor=CuArray(buckets.factor))
    output=CuArray{Float64}(undef,O,H)
    GPUResidentBatch(batch,CuArray(batch.left_values),CuArray(batch.right_values),output,paths)
end

"""Enqueue one packed product on the active stream without a host synchronization."""
function gpu_enqueue!(resident::GPUResidentBatch)
    cells=length(resident.output)
    cells==0 && return resident.output
    p=resident.paths
    @cuda threads=256 blocks=cld(cells,256) _packed_kernel!(resident.output,resident.left,
        resident.right,p.starts,p.ai,p.bi,p.factor)
    resident.output
end

"""Run one complete packed product on device; return only after synchronization."""
function gpu_run!(resident::GPUResidentBatch)
    gpu_enqueue!(resident)
    CUDA.synchronize()
    resident.output
end

"""Run resident inputs and return a distinct host-owned dense result matrix."""
gpu_owned_matrix(resident::GPUResidentBatch)=Array(gpu_run!(resident))

"""Return a fresh, owned host matrix whose memory remains CUDA-pinned while live."""
function gpu_owned_matrix_pinned(resident::GPUResidentBatch;max_host_bytes::Integer=512<<20)
    max_host_bytes>0 || throw(ArgumentError("pinned host-memory budget must be positive"))
    sizeof(resident.output)<=max_host_bytes ||
        throw(ArgumentError("pinned host output exceeds declared byte budget"))
    host=CUDA.pin(Matrix{eltype(resident.output)}(undef,size(resident.output)))
    copyto!(host,gpu_run!(resident))
    CUDA.synchronize()
    host
end

"""Include host-to-device transfer, device work and device-to-host transfer."""
gpu_complete_matrix(batch::PackedProductBatch;max_bytes::Integer=1<<30)=
    gpu_owned_matrix(gpu_resident_batch(batch;max_bytes))

"""Include packing, uploads, kernel, pinned owned-host output, and final copy."""
gpu_complete_matrix_pinned(batch::PackedProductBatch;max_bytes::Integer=1<<30,
    max_host_bytes::Integer=512<<20)=gpu_owned_matrix_pinned(
        gpu_resident_batch(batch;max_bytes);max_host_bytes)

end
