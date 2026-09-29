"""Experimental CPU threading across independent columns of an exact packed product."""
module CPUThreadedPackedPrototype
using Garamon, LinearAlgebra

export run_packed_batch_threaded, CPUResidentBatch, cpu_resident_batch,
    cpu_run_serial!, cpu_run_threaded!

struct CPUResidentBatch{B,O}
    source::B
    output::O
end

function _cpu_validate_plan(plan)
    isdiag(Garamon.metric(plan.algebra)) &&
        diag(Garamon.metric(plan.algebra))==plan.diagonal &&
        Garamon.basis(plan.algebra)==plan.basis_names ||
        throw(ArgumentError("packed product plan's algebra changed"))
    nothing
end

"""Validate once and hold a reusable output for a stable packed batch."""
function cpu_resident_batch(batch::PackedProductBatch{S};max_bytes::Integer=512<<20) where S
    max_bytes>0 || throw(ArgumentError("CPU resident byte budget must be positive"))
    plan=batch.plan
    _cpu_validate_plan(plan)
    n=size(batch.left_values,2)
    size(batch.left_values,1)==length(plan.left_masks) &&
        size(batch.right_values)==(length(plan.right_masks),n) ||
        throw(DimensionMismatch("packed values differ from plan"))
    bytes=big(sizeof(batch.left_values))+sizeof(batch.right_values)+
        big(length(plan.output_masks))*n*sizeof(S)
    bytes<=max_bytes || throw(ArgumentError("CPU resident byte budget exceeded"))
    CPUResidentBatch(batch,zeros(S,length(plan.output_masks),n))
end

"""Overwrite one resident output, preserving the original path order per column."""
function cpu_run_serial!(resident::CPUResidentBatch)
    batch=resident.source;output=resident.output;plan=batch.plan
    _cpu_validate_plan(plan)
    O=size(output,1);n=size(output,2)
    for j in 1:n
        for oi in 1:O
            @inbounds output[oi,j]=zero(eltype(output))
        end
        for (ai,bi,oi,factor) in plan.paths
            @inbounds output[oi,j]+=factor*batch.left_values[ai,j]*batch.right_values[bi,j]
        end
    end
    output
end

"""Overwrite independent resident columns using the current Julia thread pool."""
function cpu_run_threaded!(resident::CPUResidentBatch)
    batch=resident.source;output=resident.output;plan=batch.plan
    _cpu_validate_plan(plan)
    O=size(output,1);n=size(output,2)
    Threads.@threads :static for j in 1:n
        for oi in 1:O
            @inbounds output[oi,j]=zero(eltype(output))
        end
        for (ai,bi,oi,factor) in plan.paths
            @inbounds output[oi,j]+=factor*batch.left_values[ai,j]*batch.right_values[bi,j]
        end
    end
    output
end

function run_packed_batch_threaded(batch::PackedProductBatch{S}) where S
    plan=batch.plan
    _cpu_validate_plan(plan)
    n=size(batch.left_values,2)
    output=zeros(S,length(plan.output_masks),n)
    Threads.@threads :static for j in 1:n
        for (ai,bi,oi,factor) in plan.paths
            @inbounds output[oi,j]+=factor*batch.left_values[ai,j]*batch.right_values[bi,j]
        end
    end
    output
end

end
