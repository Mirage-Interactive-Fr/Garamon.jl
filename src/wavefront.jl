"""A bounded exact frontier plan for one coefficient of a product chain.

The suffix sets certify reachability by XOR of remaining blade supports. They
may overestimate reachable nonzero coefficients, but never omit one.
"""
struct WavefrontPlan{T,K,A<:GeometricAlgebra}
    algebra::A
    target::K
    supports::Vector{Vector{K}}
    suffix::Vector{Set{K}}
    max_frontier::Int
    max_pairs::Int
end

function _wavefront_same_algebra(a::GeometricAlgebra,b::GeometricAlgebra)
    dimension(a)==dimension(b) && metric(a)==metric(b) &&
        basis(a)==basis(b) && kind(a)==kind(b)
end

"""Prepare exact XOR reachability for a sparse, diagonal-metric product chain.

`target_mask` is a nonnegative blade bit mask. The suffix support budget counts
distinct masks, not paths; exceeding it throws rather than dropping a path.
"""
function prepare_wavefront(factors::AbstractVector{<:SparseMultiVector{T,K}},
                           target_mask::Integer;
                           max_suffix::Int=1<<14,
                           max_frontier::Int=1<<14,
                           max_pairs::Int=1<<20) where {T,K}
    isempty(factors) && throw(ArgumentError("wavefront needs at least one factor"))
    max_suffix>=1 && max_frontier>=1 && max_pairs>=1 ||
        throw(ArgumentError("wavefront budgets must be positive"))
    ga=first(factors).algebra
    isdiag(metric(ga)) ||
        throw(ArgumentError("wavefront requires a diagonal metric"))
    target_mask>=0 && (big(target_mask)>>dimension(ga))==0 ||
        throw(ArgumentError("target mask exceeds algebra dimension"))
    target=convert(K,target_mask)
    supports=Vector{Vector{K}}(undef,length(factors))
    for (i,factor) in enumerate(factors)
        _wavefront_same_algebra(ga,factor.algebra) ||
            throw(ArgumentError("wavefront factors belong to different algebras"))
        supports[i]=sort!(collect(keys(factor.values)))
    end
    suffix=Vector{Set{K}}(undef,length(factors)+1)
    suffix[end]=Set{K}([zero(K)])
    for i in length(factors):-1:1
        current=Set{K}()
        for a in supports[i], b in suffix[i+1]
            push!(current,a⊻b)
            length(current)<=max_suffix ||
                throw(ArgumentError("wavefront suffix support budget"))
        end
        suffix[i]=current
    end
    WavefrontPlan{T,K,typeof(ga)}(ga,target,supports,suffix,
                                  max_frontier,max_pairs)
end

"""Evaluate the planned coefficient without deleting any viable contribution.

The factor supports must match the prepared certificate. Coefficients may
change; changed support requires a new plan.
"""
function run_wavefront(plan::WavefrontPlan{T,K},
                       factors::AbstractVector{<:SparseMultiVector{T,K}}) where {T,K}
    length(factors)==length(plan.supports) ||
        throw(ArgumentError("wavefront factor count changed"))
    for (i,factor) in enumerate(factors)
        _wavefront_same_algebra(plan.algebra,factor.algebra) ||
            throw(ArgumentError("wavefront algebra changed"))
        sort!(collect(keys(factor.values)))==plan.supports[i] ||
            throw(ArgumentError("wavefront factor support changed"))
    end
    plan.target in plan.suffix[1] || return zero(T)
    frontier=Dict{K,T}(zero(K)=>one(T))
    pairs=0
    for i in eachindex(factors)
        next=Dict{K,T}()
        remaining=plan.suffix[i+1]
        for (left,leftvalue) in frontier, (right,rightvalue) in factors[i].values
            pairs+=1
            pairs<=plan.max_pairs ||
                throw(ArgumentError("wavefront pair budget"))
            mask=left⊻right
            (mask⊻plan.target) in remaining || continue
            value=_diagonal_blade_coefficient(plan.algebra,left,right,
                                               leftvalue*rightvalue)
            _addterm!(next,mask,value;max_terms=plan.max_frontier)
        end
        frontier=next
        isempty(frontier) && break
    end
    get(frontier,plan.target,zero(T))
end

"""Report the exact planned suffix widths and declared resource ceilings."""
wavefront_stats(plan::WavefrontPlan)=(
    suffix_widths=length.(plan.suffix),
    max_suffix_width=maximum(length.(plan.suffix)),
    max_frontier=plan.max_frontier,
    max_pairs=plan.max_pairs)
