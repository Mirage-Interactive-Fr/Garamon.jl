"""Exact four-column periodic residents for the resident-sum fixture contract."""
module PeriodicResidentPrototype
using Garamon, CUDA
using ..ResidentSumPrototype

export PeriodicCPUResidents, PeriodicGPUResidents, build_periodic_cpu,
    build_periodic_gpu, periodic_sum_cpu, periodic_sum_gpu_fused

struct PeriodicCPUResidents{R}
    residents::R
    horizon::Int
end

struct PeriodicGPUResidents{R}
    residents::R
    horizon::Int
end

function _periodic_fixture(fixture,horizon)
    horizon>=1 || throw(ArgumentError("periodic horizon must be positive"))
    length(fixture.inputs)>=4 ||
        throw(ArgumentError("resident fixture requires four source operand pairs"))
    # resident_sum_packs selects mod1(column+shift,4) from these exact objects,
    # so its packed operands repeat every four columns by construction.
    nothing
end

function build_periodic_cpu(fixture,horizon;max_bytes::Integer=512<<20)
    _periodic_fixture(fixture,horizon)
    residents=build_resident_sum_cpu(fixture,4;max_bytes)
    O=length(first(residents).source.plan.output_masks)
    bytes=sum(big(sizeof(r.source.left_values))+sizeof(r.source.right_values)
        for r in residents)+big(O)*horizon*sizeof(eltype(first(residents).source.left_values))
    bytes<=max_bytes || throw(ArgumentError("periodic CPU residents exceed budget"))
    PeriodicCPUResidents(residents,horizon)
end

function build_periodic_gpu(fixture,horizon;max_bytes::Integer=512<<20)
    _periodic_fixture(fixture,horizon)
    residents=build_resident_sum_gpu(fixture,4;max_bytes)
    O=length(first(residents).source.plan.output_masks)
    bytes=sum(big(sizeof(r.left))+sizeof(r.right)+sizeof(r.paths.starts)+
        sizeof(r.paths.ai)+sizeof(r.paths.bi)+sizeof(r.paths.factor)
        for r in residents)+big(O)*horizon*sizeof(Float64)
    bytes<=max_bytes || throw(ArgumentError("periodic GPU residents exceed budget"))
    PeriodicGPUResidents(residents,horizon)
end

"""Return one newly owned exact sum; every product and contribution is computed."""
function periodic_sum_cpu(packed::PeriodicCPUResidents,repetitions;threaded::Bool=false)
    repetitions>=1 || throw(ArgumentError("positive repetitions required"))
    residents=packed.residents
    length(residents)==4 || throw(ArgumentError("four periodic residents required"))
    firstbatch=first(residents).source
    O=length(firstbatch.plan.output_masks)
    H=packed.horizon
    output=zeros(eltype(firstbatch.left_values),O,H)
    for step in 1:repetitions
        batch=residents[mod1(step,4)].source
        if threaded
            Threads.@threads :static for column in 1:H
                periodcol=mod1(column,4)
                for (ai,bi,oi,factor) in batch.plan.paths
                    @inbounds output[oi,column]+=factor*
                        batch.left_values[ai,periodcol]*batch.right_values[bi,periodcol]
                end
            end
        else
            for column in 1:H
                periodcol=mod1(column,4)
                for (ai,bi,oi,factor) in batch.plan.paths
                    @inbounds output[oi,column]+=factor*
                        batch.left_values[ai,periodcol]*batch.right_values[bi,periodcol]
                end
            end
        end
    end
    output
end

@inline function _periodic_paths(value,left,right,starts,ai,bi,factor,oi,periodcol)
    @inbounds for path in starts[oi]:(starts[oi+Int32(1)]-Int32(1))
        value+=factor[path]*left[ai[path],periodcol]*right[bi[path],periodcol]
    end
    value
end

function _periodic_fused_kernel!(output,l1,r1,l2,r2,l3,r3,l4,r4,
    starts,ai,bi,factor,repetitions::Int32)
    linear=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
    linear>length(output) && return nothing
    O=Int32(size(output,1))
    oi=(linear-Int32(1))%O+Int32(1)
    column=(linear-Int32(1))÷O+Int32(1)
    periodcol=((column-Int32(1))&Int32(3))+Int32(1)
    value=zero(eltype(output))
    for step in Int32(1):repetitions
        slot=(step-Int32(1))&Int32(3)
        if slot==Int32(0)
            value=_periodic_paths(value,l1,r1,starts,ai,bi,factor,oi,periodcol)
        elseif slot==Int32(1)
            value=_periodic_paths(value,l2,r2,starts,ai,bi,factor,oi,periodcol)
        elseif slot==Int32(2)
            value=_periodic_paths(value,l3,r3,starts,ai,bi,factor,oi,periodcol)
        else
            value=_periodic_paths(value,l4,r4,starts,ai,bi,factor,oi,periodcol)
        end
    end
    @inbounds output[oi,column]=value
    nothing
end

"""One fused kernel reads four resident columns cyclically and returns an owned host result."""
function periodic_sum_gpu_fused(packed::PeriodicGPUResidents,repetitions)
    length(packed.residents)==4 && 1<=repetitions<=typemax(Int32) ||
        throw(ArgumentError("four residents and positive bounded repetitions required"))
    a,b,c,d=packed.residents
    plan=a.source.plan
    all(r->r.source.plan===plan && size(r.left,2)==4 &&
        size(r.right,2)==4,packed.residents) ||
        throw(ArgumentError("periodic resident plans or columns differ"))
    O=length(plan.output_masks)
    output=CuArray{Float64}(undef,O,packed.horizon)
    p=a.paths
    @cuda threads=256 blocks=cld(length(output),256) _periodic_fused_kernel!(output,
        a.left,a.right,b.left,b.right,c.left,c.right,d.left,d.right,
        p.starts,p.ai,p.bi,p.factor,Int32(repetitions))
    CUDA.synchronize()
    Array(output)
end

end
