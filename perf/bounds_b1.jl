module BoundsB1Prototype
using Garamon

export b1_product,b1_workspace,b1_workspace_product!,b1_workspace_product_singlepass!,B1ValidatedWorkspace

# Experimental B1 only. Both variants pay exactly the same snapshot and
# validation costs. No core method or packed kernel is replaced.
function snapshot_structure(plan::ProductPlan,a,b;allow_subsets=false)
    ga=Garamon._validate_plan_inputs(plan,a,b;allow_subsets)
    left=copy(plan.left_masks);right=copy(plan.right_masks)
    output=copy(plan.output_masks);paths=copy(plan.paths)
    for (masks,support) in ((left,plan.left_support),(right,plan.right_support))
        length(masks)==length(support) && Set(masks)==support && issorted(masks) ||
            throw(ArgumentError("B1 plan masks and support differ"))
    end
    length(unique(output))==length(output) || throw(ArgumentError("B1 duplicate output masks"))
    for masks in (left,right,output),mask in masks
        Garamon._mask_in_bounds(ga,mask) || throw(ArgumentError("B1 mask outside algebra"))
    end
    # Dense values are public mutable vectors; their constructor's length check
    # need not remain true. Keep this guard before reading any coefficients.
    for operand in (a,b)
        if operand isa DenseMultiVector
            length(operand.values)==(big(1)<<dimension(ga)) ||
                throw(DimensionMismatch("B1 dense input was resized"))
        end
    end
    A,B,O=length(left),length(right),length(output)
    for (ai,bi,oi,factor) in paths
        1<=ai<=A && 1<=bi<=B && 1<=oi<=O ||
            throw(ArgumentError("B1 path index outside private buffers"))
        # These accesses remain checked; this is not an @boundscheck block.
        output[oi]==xor(left[ai],right[bi]) ||
            throw(ArgumentError("B1 output mask differs from path XOR"))
        isfinite(factor) || throw(ArgumentError("B1 requires finite path factors"))
    end
    (;ga,left,right,output,paths)
end

function accumulate_checked!(output,left,right,paths)
    for (ai,bi,oi,factor) in paths
        output[oi]+=factor*left[ai]*right[bi]
    end
    output
end

# Internal kernel only: the caller owns all four fresh buffers/path snapshots.
# No user callbacks, mutation or suspension occur in the Float64/Int64 arithmetic.
function accumulate_inbounds!(output,left,right,paths)
    for (ai,bi,oi,factor) in paths
        @inbounds output[oi]+=factor*left[ai]*right[bi]
    end
    output
end

"""
    b1_product(plan, a, b; variant=:checked, allow_subsets=false)

Experimental local variant of the existing prepared-product algorithm. Private
copies of masks and paths are checked before any new unchecked access. Both
variants retain the original sparse/buffered branch, accumulation order and owned
sparse result. Only the buffered accumulation differs for `variant=:inbounds`.

Supports Float64/Int64 coefficients and metric factors. Callers must not modify
inputs or plan concurrently during a call. After snapshotting, the
unchecked loop touches only private vectors and immutable numeric path tuples.
Structural validation establishes memory bounds and XOR correspondence; it does
not authenticate mathematical factors or completeness of an arbitrarily forged
plan. Exact product semantics require a plan produced by `prepare_product` with
unchanged mathematical data. Copies and validation belong to total episode cost.
"""
function b1_product(plan::ProductPlan{T,K},a::AbstractMultiVector,b::AbstractMultiVector;
                    variant::Symbol=:checked,allow_subsets::Bool=false) where {T,K}
    variant in (:checked,:inbounds) || throw(ArgumentError("B1 variant must be checked or inbounds"))
    T in (Float64,Int64) && eltype(a) in (Float64,Int64) && eltype(b) in (Float64,Int64) ||
        throw(ArgumentError("B1 pilot accepts only Float64/Int64 arithmetic"))
    state=snapshot_structure(plan,a,b;allow_subsets)
    S=promote_type(T,eltype(a),eltype(b))
    if length(state.paths)<length(state.left)+length(state.right)
        result=Garamon._empty_mv(state.ga,S,:sparse)
        for (ai,bi,oi,factor) in state.paths
            Garamon._accumulate!(result,state.output[oi],
                factor*coefficient_mask(a,state.left[ai])*coefficient_mask(b,state.right[bi]))
        end
        return result
    end
    left=S[coefficient_mask(a,mask) for mask in state.left]
    right=S[coefficient_mask(b,mask) for mask in state.right]
    output=zeros(S,length(state.output))
    if variant==:inbounds
        accumulate_inbounds!(output,left,right,state.paths)
    else
        accumulate_checked!(output,left,right,state.paths)
    end
    SparseMultiVector(state.ga,Dict{K,S}(mask=>value for (mask,value) in zip(state.output,output) if !iszero(value)))
end

"""
Own a canonical, privately prepared plan and reusable coefficient buffers.
The source plan can change after construction without changing this snapshot.
Do not mutate the workspace itself or share it between concurrent calls.
Each evaluation checks the current operands and returns a new owned result.
"""
struct B1ValidatedWorkspace{S,K<:Integer,P<:ProductPlan}
    plan::P
    left_values::Vector{S}
    right_values::Vector{S}
    output_values::Vector{S}
end

function _b1_admit_workspace(canonical::ProductPlan{T,K},a,b,max_bytes) where {T,K}
    S=promote_type(T,eltype(a),eltype(b))
    workspace=B1ValidatedWorkspace{S,K,typeof(canonical)}(
        canonical,Vector{S}(undef,length(canonical.left_masks)),
        Vector{S}(undef,length(canonical.right_masks)),
        zeros(S,length(canonical.output_masks)))
    limit=min(big(max_bytes),big(Sys.free_memory())÷4)
    Base.summarysize(workspace)<=limit ||
        throw(ArgumentError("B1W exceeds retained-memory admission budget"))
    workspace
end

"""
    b1_workspace(plan,a,b;max_bytes=64<<20)

Validate the source plan against a newly generated complete canonical plan,
then retain only the canonical plan and bounded private buffers. This prototype
requires exact supports and Float64/Int64 data. `max_bytes` bounds the retained
workspace footprint at admission, not all future owned outputs or JIT memory.
"""
function b1_workspace(plan::ProductPlan{T,K},a::AbstractMultiVector,b::AbstractMultiVector;
                      max_bytes::Integer=64<<20) where {T,K}
    max_bytes>=1 || throw(ArgumentError("B1W max_bytes must be positive"))
    T in (Float64,Int64) && eltype(a) in (Float64,Int64) && eltype(b) in (Float64,Int64) ||
        throw(ArgumentError("B1W accepts only Float64/Int64 arithmetic"))
    state=snapshot_structure(plan,a,b)
    canonical=prepare_product(a,b;operation=plan.operation,
                              max_paths=big(length(state.left))*length(state.right))
    state.left==canonical.left_masks && state.right==canonical.right_masks &&
        state.output==canonical.output_masks && state.paths==canonical.paths &&
        plan.diagonal==canonical.diagonal && plan.basis_names==canonical.basis_names ||
        throw(ArgumentError("B1W source plan differs from the complete canonical product"))
    _b1_admit_workspace(canonical,a,b,max_bytes)
end

"""
    b1_workspace(a,b;operation=:geometric,max_paths=65536,max_bytes=64<<20)

Build a private canonical plan directly from the operands. The builder's
enumeration proves path indices and factors; `snapshot_structure` checks that
proof once before an unchecked reuse. Unlike the plan-taking constructor,
this route does not authenticate a separate external plan. Both routes have
the same per-call validation and owned-output contract.
"""
function b1_workspace(a::AbstractMultiVector,b::AbstractMultiVector;
                      operation::Symbol=:geometric,max_paths::Integer=1<<16,
                      max_bytes::Integer=64<<20)
    max_bytes>=1 || throw(ArgumentError("B1W max_bytes must be positive"))
    eltype(a) in (Float64,Int64) && eltype(b) in (Float64,Int64) ||
        throw(ArgumentError("B1W accepts only Float64/Int64 arithmetic"))
    ga=Garamon._same_algebra(a,b)
    for operand in (a,b)
        if operand isa DenseMultiVector
            length(operand.values)==(big(1)<<dimension(ga)) ||
                throw(DimensionMismatch("B1W dense input was resized"))
        end
        for (mask,_) in Garamon._terms(operand)
            Garamon._mask_in_bounds(ga,mask) ||
                throw(ArgumentError("B1W input mask outside algebra"))
        end
    end
    canonical=prepare_product(a,b;operation,max_paths)
    eltype(canonical.diagonal) in (Float64,Int64) ||
        throw(ArgumentError("B1W accepts only Float64/Int64 metric factors"))
    snapshot_structure(canonical,a,b)
    _b1_admit_workspace(canonical,a,b,max_bytes)
end

"""
    b1_workspace_product!(workspace,a,b)

Reuse only the private buffers and validated canonical paths. The operands are
rechecked on every call; changed coefficients are read on every call. The
returned sparse multivector owns a fresh dictionary. A source-plan mutation is
irrelevant to this snapshot; a metric, basis, support or numeric-type change in
the operands is rejected before the unchecked accumulation.
"""
function _b1_workspace_product!(workspace::B1ValidatedWorkspace{S,K},
                                a::AbstractMultiVector,b::AbstractMultiVector,
                                ::Val{materialization}) where {S,K,materialization}
    plan=workspace.plan
    promote_type(eltype(plan.diagonal),eltype(a),eltype(b))==S ||
        throw(ArgumentError("B1W numeric type differs from its snapshot"))
    ga=Garamon._validate_plan_inputs(plan,a,b)
    for operand in (a,b)
        if operand isa DenseMultiVector
            length(operand.values)==(big(1)<<dimension(ga)) ||
                throw(DimensionMismatch("B1W dense input was resized"))
        end
    end
    for (i,mask) in enumerate(plan.left_masks)
        workspace.left_values[i]=coefficient_mask(a,mask)
    end
    for (i,mask) in enumerate(plan.right_masks)
        workspace.right_values[i]=coefficient_mask(b,mask)
    end
    fill!(workspace.output_values,zero(S))
    accumulate_inbounds!(workspace.output_values,workspace.left_values,
                         workspace.right_values,plan.paths)
    if materialization===:singlepass
        # The empty public constructor validates the descriptor and owns a fresh
        # dictionary. Canonical private output masks were checked at admission.
        # They are unique, so direct writes reproduce the constructor result.
        result=Garamon._empty_mv(ga,S,:sparse)::SparseMultiVector{S,K}
        for (mask,value) in zip(plan.output_masks,workspace.output_values)
            iszero(value) || (result.values[mask]=value)
        end
        return result
    end
    SparseMultiVector(ga,Dict{K,S}(mask=>value for (mask,value) in
        zip(plan.output_masks,workspace.output_values) if !iszero(value)))
end

b1_workspace_product!(workspace::B1ValidatedWorkspace,a::AbstractMultiVector,b::AbstractMultiVector)=
    _b1_workspace_product!(workspace,a,b,Val(:constructor))

"""Experimental B1W output construction with a single dictionary insertion pass."""
b1_workspace_product_singlepass!(workspace::B1ValidatedWorkspace,
                                 a::AbstractMultiVector,b::AbstractMultiVector)=
    _b1_workspace_product!(workspace,a,b,Val(:singlepass))
end
