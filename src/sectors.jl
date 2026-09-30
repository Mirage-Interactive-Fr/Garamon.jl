"""Prepared even/odd grade sectors for an exact geometric product."""
struct InvariantSectorPlan{P<:ProductPlan}
    plans::NTuple{4,P}  # even-even, even-odd, odd-even, odd-odd
    output_parity::Symbol
end

_sector_keytype(::DenseMultiVector)=UInt64
_sector_keytype(::SparseMultiVector{T,K}) where {T,K}=K

function _sector_parts(a::AbstractMultiVector)
    K=_sector_keytype(a)
    even=Dict{K,BigInt}()
    odd=Dict{K,BigInt}()
    for (mask,value) in _terms(a)
        target=iseven(_blade_grade(mask)) ? even : odd
        target[convert(K,mask)]=BigInt(value)
    end
    SparseMultiVector(a.algebra,even),SparseMultiVector(a.algebra,odd)
end

function _sector_projection(a::AbstractMultiVector, parity::Symbol)
    parity==:all && return a
    K=_sector_keytype(a)
    SparseMultiVector(a.algebra,Dict{K,BigInt}(
        convert(K,mask)=>BigInt(value) for (mask,value) in _terms(a)
        if iseven(_blade_grade(mask))==(parity==:even)))
end

"""
    prepare_invariant_sectors(a,b; output_parity=:all,max_paths=65536)

Prepare the eigenspaces of grade involution. For a diagonal metric,
parity(a*b)=parity(a) xor parity(b), so a selected output parity requires
only two of the four sector products. Preparation fixes both input supports.
"""
function prepare_invariant_sectors(a::AbstractMultiVector,b::AbstractMultiVector;
        output_parity::Symbol=:all,max_paths::Integer=1<<16)
    ga=_same_algebra(a,b)
    output_parity in (:all,:even,:odd) ||
        throw(ArgumentError("output_parity must be :all, :even, or :odd"))
    eltype(a)<:Integer && eltype(b)<:Integer &&
        eltype(metric(ga))<:Integer && isdiag(metric(ga)) &&
        all(metric(ga)[i,i] in (-1,0,1) for i in 1:dimension(ga)) ||
        throw(ArgumentError("invariant sectors require integer inputs and a diagonal metric in {-1,0,1}"))
    ae,ao=_sector_parts(a)
    be,bo=_sector_parts(b)
    plans=(prepare_product(ae,be;max_paths),
        prepare_product(ae,bo;max_paths),
        prepare_product(ao,be;max_paths),
        prepare_product(ao,bo;max_paths))
    InvariantSectorPlan(plans,output_parity)
end

"""Execute a prepared sector product, with an exact direct fallback if requested."""
function run_invariant_sectors(plan::InvariantSectorPlan,
        a::AbstractMultiVector,b::AbstractMultiVector;
        on_invalid::Symbol=:error,diagnostics::Bool=false)
    on_invalid in (:error,:direct) ||
        throw(ArgumentError("on_invalid must be :error or :direct"))
    ga=_same_algebra(a,b)
    eltype(a)<:Integer && eltype(b)<:Integer &&
        eltype(metric(ga))<:Integer ||
        throw(ArgumentError("exact sector execution requires integer inputs and metric"))
    ae,ao=_sector_parts(a)
    be,bo=_sector_parts(b)
    pairs=((ae,be),(ae,bo),(ao,be),(ao,bo))
    valid=try
        for i in eachindex(plan.plans)
            _validate_plan_inputs(plan.plans[i],pairs[i]...)
        end
        true
    catch exception
        exception isa ArgumentError || rethrow()
        false
    end
    if !valid
        on_invalid==:error && throw(ArgumentError("invariant sector plan is invalid"))
        result=_sector_projection(geometric_product(_certificate_bigint(a),
            _certificate_bigint(b)),plan.output_parity)
        return diagnostics ? (result,(used_fallback=true,selected_pairs=4)) : result
    end
    selected=plan.output_parity==:even ? (1,4) :
        plan.output_parity==:odd ? (2,3) : (1,2,3,4)
    K=eltype(plan.plans[1].left_masks)
    result=SparseMultiVector(ga,Dict{K,BigInt}())
    for i in selected
        partial=run_product(plan.plans[i],pairs[i]...)
        for (mask,value) in _terms(partial)
            _accumulate!(result,mask,value)
        end
    end
    diagnostics ? (result,(used_fallback=false,selected_pairs=length(selected))) : result
end
