"""A reusable structural plan for an exact integer, diagonal-metric product.

The residues are reconstructed only after their combined modulus exceeds
twice a coefficient bound. Otherwise `run_modular_product` uses its exact
integer fallback or raises an error, according to the requested contract.
"""
struct ModularProductPlan{K<:Integer,A<:GeometricAlgebra}
    algebra::A
    diagonal::Vector{BigInt}
    basis_names::Vector{String}
    left_masks::Vector{K}
    right_masks::Vector{K}
    output_masks::Vector{K}
    paths::Vector{Tuple{Int,Int,Int,BigInt}}
    primes::Vector{Int64}
end

function _modular_prime(candidate::Int64)
    candidate>=2 || return false
    candidate==2 && return true
    iseven(candidate) && return false
    divisor=Int64(3)
    while divisor<=div(candidate,divisor)
        candidate%divisor==0 && return false
        divisor+=2
    end
    true
end

function _modular_primes(bits::Integer,count::Integer)
    17<=bits<=31 || throw(ArgumentError("prime_bits must be in [17,31]"))
    1<=count<=64 || throw(ArgumentError("max_primes must be in [1,64]"))
    result=Int64[]
    candidate=(Int64(1)<<bits)-1
    iseven(candidate) && (candidate-=1)
    while length(result)<count
        _modular_prime(candidate) && push!(result,candidate)
        candidate-=2
        candidate>2 || error("prime search exhausted")
    end
    result
end

"""
    prepare_modular_product(a, b; prime_bits=31, max_primes=8,
                            max_paths=65536, max_terms=65536)

Prepare exact structural paths and pairwise-coprime prime moduli. Inputs must
have integer coefficients and belong to an algebra with a diagonal integer
metric. The path and output budgets are checked before constructing the plan.
"""
function prepare_modular_product(a::AbstractMultiVector,b::AbstractMultiVector;
        prime_bits::Integer=31,max_primes::Integer=8,
        max_paths::Integer=1<<16,max_terms::Integer=1<<16)
    ga=_same_algebra(a,b)
    eltype(a)<:Integer && eltype(b)<:Integer &&
        eltype(metric(ga))<:Integer && isdiag(metric(ga)) ||
        throw(ArgumentError("modular product requires integer coefficients and a diagonal integer metric"))
    max_paths>=0 && max_terms>=1 || throw(ArgumentError("invalid modular product budget"))
    K=_masktype(ga)
    left_masks=sort!(K[mask for (mask,_) in _terms(a)])
    right_masks=sort!(K[mask for (mask,_) in _terms(b)])
    big(length(left_masks))*length(right_masks)<=max_paths ||
        throw(ArgumentError("modular product exceeds max_paths"))
    diagonal=BigInt[metric(ga)[i,i] for i in 1:dimension(ga)]
    output_masks=K[]
    output_index=Dict{K,Int}()
    paths=Tuple{Int,Int,Int,BigInt}[]
    for (ai,amask) in enumerate(left_masks),(bi,bmask) in enumerate(right_masks)
        factor=BigInt(_shuffle_sign(ga,amask,bmask))
        overlap=amask & bmask
        while !iszero(overlap)
            factor*=diagonal[trailing_zeros(overlap)+1]
            iszero(factor) && break
            overlap &= overlap-one(overlap)
        end
        iszero(factor) && continue
        output=amask ⊻ bmask
        oi=get!(output_index,output) do
            length(output_masks)<max_terms ||
                throw(ArgumentError("modular product exceeds max_terms"))
            push!(output_masks,output)
            length(output_masks)
        end
        push!(paths,(ai,bi,oi,factor))
    end
    ModularProductPlan(ga,diagonal,copy(basis(ga)),left_masks,right_masks,
        output_masks,paths,_modular_primes(prime_bits,max_primes))
end

function _validate_modular_inputs(plan::ModularProductPlan,a,b)
    ga=_same_algebra(a,b)
    dimension(ga)==length(plan.diagonal) && isdiag(metric(ga)) &&
        kind(ga)==kind(plan.algebra) && basis(ga)==plan.basis_names &&
        all(metric(ga)[i,i]==plan.diagonal[i] for i in eachindex(plan.diagonal)) ||
        throw(ArgumentError("modular plan belongs to another algebra or metric"))
    eltype(a)<:Integer && eltype(b)<:Integer ||
        throw(ArgumentError("modular execution requires integer coefficients"))
    sort!([mask for (mask,_) in _terms(a)])==plan.left_masks &&
        sort!([mask for (mask,_) in _terms(b)])==plan.right_masks ||
        throw(ArgumentError("modular plan input support changed"))
    ga
end

function _modular_output(plan::ModularProductPlan{K},ga,values) where K
    SparseMultiVector(ga,Dict{K,BigInt}(mask=>value
        for (mask,value) in zip(plan.output_masks,values) if !iszero(value)))
end

"""
    run_modular_product(plan, a, b; fallback=true, diagnostics=false)

Return the exact integer geometric product. With `diagnostics=true`, return
`(product, (prime_count, used_fallback, bound_bits))`. CRT reconstruction is
used only when its modulus is greater than twice the maximum absolute
coefficient bound. A requested `fallback=false` rejects insufficient moduli.
"""
function run_modular_product(plan::ModularProductPlan,a::AbstractMultiVector,
        b::AbstractMultiVector;fallback::Bool=true,diagnostics::Bool=false)
    ga=_validate_modular_inputs(plan,a,b)
    left=BigInt[coefficient_mask(a,mask) for mask in plan.left_masks]
    right=BigInt[coefficient_mask(b,mask) for mask in plan.right_masks]
    bounds=zeros(BigInt,length(plan.output_masks))
    for (ai,bi,oi,factor) in plan.paths
        bounds[oi]+=abs(left[ai])*abs(right[bi])*abs(factor)
    end
    bound=isempty(bounds) ? BigInt(0) : maximum(bounds)
    bound_bits=iszero(bound) ? 0 : ndigits(bound;base=2)
    modulus=BigInt(1)
    prime_count=0
    while prime_count<length(plan.primes) && modulus<=2*bound
        prime_count+=1
        modulus*=plan.primes[prime_count]
    end
    if modulus<=2*bound
        fallback || throw(ArgumentError("CRT modulus does not certify exact reconstruction"))
        values=zeros(BigInt,length(plan.output_masks))
        for (ai,bi,oi,factor) in plan.paths
            values[oi]+=left[ai]*right[bi]*factor
        end
        product=_modular_output(plan,ga,values)
        info=(prime_count=0,used_fallback=true,bound_bits=bound_bits)
        return diagnostics ? (product,info) : product
    end
    reconstructed=zeros(BigInt,length(plan.output_masks))
    previous_modulus=BigInt(1)
    for prime in plan.primes[1:prime_count]
        left_residues=Int64[mod(value,prime) for value in left]
        right_residues=Int64[mod(value,prime) for value in right]
        residues=zeros(Int64,length(plan.output_masks))
        for (ai,bi,oi,factor) in plan.paths
            contribution=mod(Int128(left_residues[ai])*right_residues[bi],prime)
            contribution=mod(contribution*Int64(mod(factor,prime)),prime)
            residues[oi]=Int64(mod(Int128(residues[oi])+contribution,prime))
        end
        inverse=invmod(Int64(mod(previous_modulus,prime)),prime)
        for i in eachindex(reconstructed)
            difference=mod(Int128(residues[i])-Int64(mod(reconstructed[i],prime)),prime)
            correction=mod(difference*inverse,prime)
            reconstructed[i]+=previous_modulus*correction
        end
        previous_modulus*=prime
    end
    half=div(previous_modulus,2)
    for i in eachindex(reconstructed)
        reconstructed[i]>half && (reconstructed[i]-=previous_modulus)
        abs(reconstructed[i])<=bounds[i] ||
            error("CRT reconstruction violated its coefficient bound")
    end
    product=_modular_output(plan,ga,reconstructed)
    info=(prime_count=prime_count,used_fallback=false,bound_bits=bound_bits)
    diagnostics ? (product,info) : product
end

"""Prepare and execute one exact multimodular product."""
function modular_geometric_product(a::AbstractMultiVector,b::AbstractMultiVector;
        prime_bits::Integer=31,max_primes::Integer=8,max_paths::Integer=1<<16,
        max_terms::Integer=1<<16,fallback::Bool=true,diagnostics::Bool=false)
    plan=prepare_modular_product(a,b;prime_bits,max_primes,max_paths,max_terms)
    run_modular_product(plan,a,b;fallback,diagnostics)
end
