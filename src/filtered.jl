"""A structural plan for one exact integer product-coefficient sign."""
struct FilteredSignPlan{K<:Integer,A<:GeometricAlgebra}
    algebra::A
    diagonal::Vector{BigInt}
    basis_names::Vector{String}
    target::K
    left_masks::Vector{K}
    right_masks::Vector{K}
    paths::Vector{Tuple{Int,Int,BigInt}}
end

"""
    prepare_filtered_sign(a,b,indices; max_paths=65536)

Prepare every path to one coefficient of the geometric product. Requires an
integer diagonal metric and integer coefficients. The checked fast path is
only an execution choice; the requested sign is always exact.
"""
function prepare_filtered_sign(a::AbstractMultiVector,b::AbstractMultiVector,
        indices;max_paths::Integer=1<<16)
    ga=_same_algebra(a,b)
    eltype(a)<:Integer && eltype(b)<:Integer &&
        eltype(metric(ga))<:Integer && isdiag(metric(ga)) ||
        throw(ArgumentError("filtered sign requires integer coefficients and a diagonal integer metric"))
    max_paths>=0 || throw(ArgumentError("max_paths must be nonnegative"))
    K=_masktype(ga)
    target=_mask(ga,indices,K)
    left_masks=sort!(K[mask for (mask,_) in _terms(a)])
    right_masks=sort!(K[mask for (mask,_) in _terms(b)])
    right_index=Dict(mask=>i for (i,mask) in enumerate(right_masks))
    diagonal=BigInt[metric(ga)[i,i] for i in 1:dimension(ga)]
    paths=Tuple{Int,Int,BigInt}[]
    for (ai,amask) in enumerate(left_masks)
        bmask=amask⊻target
        bi=get(right_index,bmask,0)
        iszero(bi) && continue
        factor=BigInt(_shuffle_sign(ga,amask,bmask))
        overlap=amask&bmask
        while !iszero(overlap)
            factor*=diagonal[trailing_zeros(overlap)+1]
            iszero(factor) && break
            overlap &= overlap-one(overlap)
        end
        iszero(factor) && continue
        length(paths)<max_paths ||
            throw(ArgumentError("filtered sign exceeds max_paths"))
        push!(paths,(ai,bi,factor))
    end
    FilteredSignPlan(ga,diagonal,copy(basis(ga)),target,left_masks,
        right_masks,paths)
end

function _filtered_validate(plan::FilteredSignPlan,a,b)
    ga=_same_algebra(a,b)
    dimension(ga)==length(plan.diagonal) && isdiag(metric(ga)) &&
        kind(ga)==kind(plan.algebra) && basis(ga)==plan.basis_names &&
        all(metric(ga)[i,i]==plan.diagonal[i] for i in eachindex(plan.diagonal)) ||
        throw(ArgumentError("filtered sign plan belongs to another algebra"))
    eltype(a)<:Integer && eltype(b)<:Integer ||
        throw(ArgumentError("filtered sign requires integer coefficients"))
    sort!([mask for (mask,_) in _terms(a)])==plan.left_masks &&
        sort!([mask for (mask,_) in _terms(b)])==plan.right_masks ||
        throw(ArgumentError("filtered sign input support changed"))
    ga
end

"""
    run_filtered_sign(plan,a,b; diagnostics=false)

Return -1, 0, or 1. Checked Int128 arithmetic certifies the fast result;
any overflow or conversion failure triggers exact BigInt recomputation.
"""
function run_filtered_sign(plan::FilteredSignPlan,a::AbstractMultiVector,
        b::AbstractMultiVector;diagnostics::Bool=false)
    _filtered_validate(plan,a,b)
    left=[coefficient_mask(a,mask) for mask in plan.left_masks]
    right=[coefficient_mask(b,mask) for mask in plan.right_masks]
    fallback=false
    result=try
        value=Int128(0)
        for (ai,bi,factor) in plan.paths
            term=Base.Checked.checked_mul(Int128(left[ai]),Int128(right[bi]))
            term=Base.Checked.checked_mul(term,Int128(factor))
            value=Base.Checked.checked_add(value,term)
        end
        Int8(sign(value))
    catch exception
        exception isa OverflowError || exception isa InexactError || rethrow()
        fallback=true
        value=BigInt(0)
        for (ai,bi,factor) in plan.paths
            value+=BigInt(left[ai])*BigInt(right[bi])*factor
        end
        Int8(sign(value))
    end
    diagnostics ? (result,(used_fallback=fallback,path_count=length(plan.paths))) :
        result
end

"""One-shot exact filtered sign of a targeted geometric-product coefficient."""
function filtered_product_sign(a::AbstractMultiVector,b::AbstractMultiVector,
        indices;max_paths::Integer=1<<16,diagnostics::Bool=false)
    plan=prepare_filtered_sign(a,b,indices;max_paths)
    run_filtered_sign(plan,a,b;diagnostics)
end
