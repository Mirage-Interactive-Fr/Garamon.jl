"""Experimental B4: bounds-proof comparison for the compact binary-rank kernel."""
module BoundsB4Prototype
using Garamon
using LinearAlgebra
using ..BinaryRankCompactPrototype: CompactRankWorkspace, prepare_compact_rank,
    validate_compact, accumulate_compact!, owned_compact_result

export B4ValidatedWorkspace, b4_workspace, b4_product!

struct B4ValidatedWorkspace{W}
    inner::W
end

function _check_b4_structure(plan,workspace)
    A=length(plan.left_masks);B=length(plan.right_masks);O=length(plan.phi)
    size(plan.output_index)==(A,B) && size(plan.cocycle)==(A,B) ||
        throw(DimensionMismatch("B4 index/cocycle matrix shape differs from support"))
    length(workspace.left_values)==A && length(workspace.right_values)==B &&
        length(workspace.output_values)==O ||
        throw(DimensionMismatch("B4 numerical buffer shape differs from plan"))
    all(oi->oi==0 || 1<=oi<=O,plan.output_index) ||
        throw(ArgumentError("B4 output index outside numerical buffer"))
    length(unique(plan.phi))==O &&
        all(mask->Garamon._mask_in_bounds(plan.algebra,mask),plan.phi) ||
        throw(ArgumentError("B4 output masks must be unique and inside the algebra"))
    all(isfinite,plan.cocycle) || throw(ArgumentError("B4 nonfinite cocycle"))
    nothing
end

"""
    b4_workspace(a,b; query=(:all,0), max_bytes=64<<20)

Build a canonical compact-rank plan directly from sparse operands and verify
all index/buffer bounds once before reuse. Inputs are checked on every product.
Neither the workspace nor its contained plan may be mutated or shared across
concurrent evaluations. The bound covers retained plan and buffers, not outputs
or JIT memory. No contributions are omitted beyond the exact requested query.
"""
function b4_workspace(a::SparseMultiVector,b::SparseMultiVector;
                      query=(:all,0),max_rank::Integer=16,
                      max_pairs::Integer=65_536,max_outputs::Integer=65_536,
                      max_bytes::Integer=64<<20)
    max_bytes>=1 || throw(ArgumentError("B4 memory budget must be positive"))
    eltype(a) in (Float64,Int64) && eltype(b) in (Float64,Int64) ||
        throw(ArgumentError("B4 pilot accepts Float64/Int64 coefficients"))
    ga=Garamon._same_algebra(a,b)
    all(isfinite,diag(Garamon.metric(ga))) ||
        throw(ArgumentError("B4 requires a finite diagonal metric"))
    for mv in (a,b),mask in keys(mv.values)
        Garamon._mask_in_bounds(ga,mask) ||
            throw(ArgumentError("B4 input mask outside algebra"))
    end
    plan=prepare_compact_rank(a,b;query,max_rank,max_pairs,max_outputs,max_bytes)
    S=promote_type(eltype(plan.diagonal),eltype(a),eltype(b))
    S in (Float64,Int64) || throw(ArgumentError("B4 pilot accepts Float64/Int64 arithmetic"))
    workspace=CompactRankWorkspace(plan,S;max_bytes)
    _check_b4_structure(plan,workspace)
    Base.summarysize(workspace)<=min(big(max_bytes),big(Sys.free_memory())÷4) ||
        throw(ArgumentError("B4 retained workspace exceeds available-memory admission"))
    B4ValidatedWorkspace(workspace)
end

# Only canonical privately prepared matrices and size-matched buffers enter
# this loop. The operand dictionaries are read by key; their support is checked
# before this call. The numerical order matches accumulate_compact! exactly.
function _accumulate_b4_inbounds!(out,av,bv,plan,a,b)
    fill!(out,zero(eltype(out)))
    @inbounds for i in eachindex(av)
        av[i]=get(a.values,plan.left_masks[i],zero(eltype(av)))
    end
    @inbounds for j in eachindex(bv)
        bv[j]=get(b.values,plan.right_masks[j],zero(eltype(bv)))
    end
    @inbounds for j in eachindex(bv),i in eachindex(av)
        oi=plan.output_index[i,j]
        iszero(oi) && continue
        out[oi]+=plan.cocycle[i,j]*av[i]*bv[j]
    end
    out
end

"""Compare one checked or locally annotated compact-rank evaluation."""
function b4_product!(workspace::B4ValidatedWorkspace,
                     a::SparseMultiVector,b::SparseMultiVector;
                     variant::Symbol=:checked,allow_subsets::Bool=false)
    variant in (:checked,:inbounds) || throw(ArgumentError("B4 variant must be checked or inbounds"))
    inner=workspace.inner;plan=inner.plan
    validate_compact(plan,a,b;allow_subsets)
    S=eltype(inner.output_values)
    promote_type(eltype(plan.diagonal),eltype(a),eltype(b))==S ||
        throw(ArgumentError("B4 coefficient type differs from workspace"))
    if variant===:inbounds
        _accumulate_b4_inbounds!(inner.output_values,inner.left_values,
                                inner.right_values,plan,a,b)
    else
        accumulate_compact!(inner.output_values,inner.left_values,
                            inner.right_values,plan,a,b)
    end
    owned_compact_result(plan,inner.output_values,a.algebra)
end
end
