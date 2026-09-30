"""A target-ordered exact exterior-product plan for bounded dimensions."""
struct SignedDisjointPlan{A<:GeometricAlgebra,T<:Number}
    algebra::A
    metric_snapshot::Matrix{T}
    basis_names::Vector{String}
    left_masks::Vector{UInt64}
    right_masks::Vector{UInt64}
    output_masks::Vector{UInt64}
    paths::Vector{Vector{Tuple{Int,Int,Int8}}}
    max_terms::Int
end

function _disjoint_domain(a,b,max_dimension,max_paths,max_terms)
    ga=_same_algebra(a,b)
    2<=dimension(ga)<=min(max_dimension,12) ||
        throw(ArgumentError("signed disjoint scan supports dimensions 2 through 12"))
    eltype(a)<:Integer && eltype(b)<:Integer ||
        throw(ArgumentError("signed disjoint scan requires integer coefficients"))
    max_paths>=1 && max_terms>=1 ||
        throw(ArgumentError("invalid disjoint convolution budget"))
    big(3)^dimension(ga)<=max_paths ||
        throw(ArgumentError("disjoint target scan exceeds max_paths"))
    ga
end

"""
    prepare_signed_disjoint_convolution(a,b; max_dimension=12,
                                        max_paths=1_000_000,max_terms=65536)

Enumerate each target mask and its disjoint partitions once, retaining only
paths supported by both inputs. The structural path count is bounded before
enumeration; input support must remain the same at execution.
"""
function prepare_signed_disjoint_convolution(a::AbstractMultiVector,
        b::AbstractMultiVector;max_dimension::Integer=12,
        max_paths::Integer=1_000_000,max_terms::Integer=1<<16)
    ga=_disjoint_domain(a,b,max_dimension,max_paths,max_terms)
    left_masks=sort!(UInt64[mask for (mask,_) in _terms(a)])
    right_masks=sort!(UInt64[mask for (mask,_) in _terms(b)])
    left_index=Dict(mask=>i for (i,mask) in enumerate(left_masks))
    right_index=Dict(mask=>i for (i,mask) in enumerate(right_masks))
    output_masks=UInt64[]
    paths=Vector{Tuple{Int,Int,Int8}}[]
    for target in UInt64(0):(UInt64(1)<<dimension(ga))-UInt64(1)
        group=Tuple{Int,Int,Int8}[]
        submask=target
        while true
            other=target⊻submask
            ai=get(left_index,submask,0)
            bi=get(right_index,other,0)
            if ai!=0 && bi!=0
                push!(group,(ai,bi,Int8(_shuffle_sign(ga,submask,other))))
            end
            iszero(submask) && break
            submask=(submask-UInt64(1))&target
        end
        if !isempty(group)
            length(output_masks)<max_terms ||
                throw(ArgumentError("disjoint output exceeds max_terms"))
            push!(output_masks,target)
            push!(paths,group)
        end
    end
    SignedDisjointPlan(ga,Matrix(metric(ga)),copy(basis(ga)),left_masks,
        right_masks,output_masks,paths,Int(max_terms))
end

"""Run a signed disjoint plan with exact BigInt accumulation."""
function run_signed_disjoint_convolution(plan::SignedDisjointPlan,
        a::AbstractMultiVector,b::AbstractMultiVector)
    ga=_same_algebra(a,b)
    dimension(ga)==dimension(plan.algebra) &&
        kind(ga)==kind(plan.algebra) && basis(ga)==plan.basis_names &&
        metric(ga)==plan.metric_snapshot ||
        throw(ArgumentError("disjoint plan belongs to another algebra"))
    eltype(a)<:Integer && eltype(b)<:Integer ||
        throw(ArgumentError("disjoint inputs must have integer coefficients"))
    sort!(UInt64[mask for (mask,_) in _terms(a)])==plan.left_masks &&
        sort!(UInt64[mask for (mask,_) in _terms(b)])==plan.right_masks ||
        throw(ArgumentError("disjoint plan input support changed"))
    left=BigInt[coefficient_mask(a,mask) for mask in plan.left_masks]
    right=BigInt[coefficient_mask(b,mask) for mask in plan.right_masks]
    output=Dict{UInt64,BigInt}()
    for (target,group) in zip(plan.output_masks,plan.paths)
        value=BigInt(0)
        for (ai,bi,sign) in group
            value+=sign*left[ai]*right[bi]
        end
        iszero(value) || (output[target]=value)
    end
    SparseMultiVector(ga,output)
end

"""One-shot target-ordered exact signed disjoint convolution."""
function signed_disjoint_convolution(a::AbstractMultiVector,b::AbstractMultiVector;
        max_dimension::Integer=12,max_paths::Integer=1_000_000,
        max_terms::Integer=1<<16)
    plan=prepare_signed_disjoint_convolution(a,b;
        max_dimension,max_paths,max_terms)
    run_signed_disjoint_convolution(plan,a,b)
end

