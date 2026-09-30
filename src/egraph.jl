"""A bounded associativity equivalence class for an exact product chain.

Every interval contains all binary splits of the ordered factors. The extractor
chooses the split with the lowest support-pair upper bound. This restricted
e-graph deliberately has no commutativity or floating-point rewrite rules.
"""
struct ProductEGraphPlan{A<:GeometricAlgebra}
    algebra::A
    supports::Vector{Vector{UInt64}}
    splits::Matrix{Int}
    estimated_pairs::Int
    candidates::Int
    max_terms::Int
end

function _egraph_same_algebra(a::GeometricAlgebra,b::GeometricAlgebra)
    dimension(a)==dimension(b) && metric(a)==metric(b) &&
        basis(a)==basis(b) && kind(a)==kind(b)
end

function _egraph_support_product(left::Vector{UInt64},right::Vector{UInt64},
                                 null_mask::UInt64,max_support::Int)
    result=Set{UInt64}()
    for a in left,b in right
        iszero(a & b & null_mask) && push!(result,xor(a,b))
        length(result)<=max_support ||
            throw(ArgumentError("e-graph support budget exceeded"))
    end
    sort!(collect(result))
end

"""Saturate contiguous associativity splits and extract a support-cost plan.

Only sparse BigInt inputs over a diagonal integral metric are admitted. A plan
may be reused with new coefficients while every input support remains fixed.
"""
function prepare_product_egraph(factors::AbstractVector{<:SparseMultiVector{BigInt}};
                                max_candidates::Int=64,max_support::Int=1<<16,
                                max_terms::Int=1<<16)
    count=length(factors)
    2<=count<=8 || throw(ArgumentError("e-graph needs two to eight factors"))
    max_candidates>=1 && max_support>=1 && max_terms>=1 ||
        throw(ArgumentError("e-graph budgets must be positive"))
    ga=first(factors).algebra
    dimension(ga)<=64 && isdiag(metric(ga)) &&
        eltype(metric(ga))<:Integer ||
        throw(ArgumentError("e-graph needs a diagonal integral metric up to dimension 64"))
    all(f->_egraph_same_algebra(ga,f.algebra),factors) ||
        throw(ArgumentError("e-graph factor algebra mismatch"))
    null_mask=foldl(|,(UInt64(1)<<(i-1) for i in 1:dimension(ga)
                       if iszero(metric(ga)[i,i]));init=UInt64(0))
    supports=[sort!(collect(keys(f.values))) for f in factors]
    interval=Matrix{Vector{UInt64}}(undef,count,count)
    cost=fill(typemax(Int),count,count)
    splits=zeros(Int,count,count)
    candidates=0
    for i in 1:count
        interval[i,i]=supports[i]
        cost[i,i]=0
    end
    for width in 2:count, i in 1:count-width+1
        j=i+width-1
        # The support bound is associative; any split produces the same set.
        interval[i,j]=_egraph_support_product(interval[i,j-1],supports[j],
                                              null_mask,max_support)
        for k in i:j-1
            candidates+=1
            candidates<=max_candidates ||
                throw(ArgumentError("e-graph candidate budget exceeded"))
            pair_count=Base.checked_mul(length(interval[i,k]),
                                        length(interval[k+1,j]))
            estimate=Base.checked_add(Base.checked_add(cost[i,k],cost[k+1,j]),
                                      pair_count)
            if estimate<cost[i,j]
                cost[i,j]=estimate
                splits[i,j]=k
            end
        end
    end
    ProductEGraphPlan(ga,supports,splits,cost[1,count],candidates,max_terms)
end

function _run_product_egraph(plan::ProductEGraphPlan,factors,i::Int,j::Int)
    i==j && return factors[i]
    k=plan.splits[i,j]
    geometric_product(_run_product_egraph(plan,factors,i,k),
                      _run_product_egraph(plan,factors,k+1,j);
                      max_terms=plan.max_terms)
end

"""Evaluate an extracted exact parenthesization on matching supports."""
function run_product_egraph(plan::ProductEGraphPlan,
                            factors::AbstractVector{<:SparseMultiVector{BigInt}})
    length(factors)==length(plan.supports) ||
        throw(ArgumentError("e-graph factor count changed"))
    for (index,factor) in enumerate(factors)
        _egraph_same_algebra(plan.algebra,factor.algebra) &&
            sort!(collect(keys(factor.values)))==plan.supports[index] ||
            throw(ArgumentError("e-graph algebra or support changed"))
    end
    _run_product_egraph(plan,factors,1,length(factors))
end

product_egraph_stats(plan::ProductEGraphPlan)=(
    estimated_pairs=plan.estimated_pairs,candidates=plan.candidates,
    root_split=plan.splits[1,length(plan.supports)])
