"""A thread-safe cache with a byte budget for retained structural plans.

Policy metadata is reported separately by `cache_stats` and is not part of
`max_bytes`. TinyLFU combines a bounded decaying frequency sketch with LRU
replacement; SIEVE uses a FIFO queue and visited bits. Neither policy skips
any contribution to a requested algebraic product.
"""
mutable struct ProductPlanCache
    max_bytes::Int
    policy::Symbol
    eviction_exponent::Float64
    used_bytes::Int
    plans::Dict{Any,ProductPlan}
    sizes::Dict{Any,Int}
    ages::Dict{Any,UInt64}
    hit_counts::Dict{Any,Int}
    rng::Random.Xoshiro
    clock::UInt64
    hits::Int
    misses::Int
    evictions::Int
    multi_candidate_evictions::Int
    frequency::Matrix{UInt8}
    frequency_samples::Int
    admissions_rejected::Int
    sieve_order::Vector{Any}
    sieve_visited::Dict{Any,Bool}
    sieve_hand::Int
    sieve_scans::Int
    lock::ReentrantLock
    function ProductPlanCache(; max_bytes::Integer=1 << 20,
                              policy::Symbol=:lru,
                              eviction_exponent::Real=1.0,
                              frequency_slots::Integer=256,
                              seed::Integer=0x676172616d6f6e)
        0 <= max_bytes <= typemax(Int) ||
            throw(ArgumentError("cache byte budget must be a nonnegative Int"))
        policy in (:lru, :roulette, :tinylfu, :sieve) ||
            throw(ArgumentError("cache policy must be :lru, :roulette, :tinylfu or :sieve"))
        isfinite(eviction_exponent) && 0 <= eviction_exponent <= 8 ||
            throw(ArgumentError("cache eviction exponent must be finite in [0,8]"))
        0 <= seed <= typemax(UInt64) ||
            throw(ArgumentError("cache seed must fit UInt64"))
        frequency_slots isa Int && 16 <= frequency_slots <= 1 << 20 &&
            ispow2(frequency_slots) ||
            throw(ArgumentError("frequency slots must be a power of two in [16, 2^20]"))
        new(Int(max_bytes), policy, Float64(eviction_exponent), 0,
            Dict{Any,ProductPlan}(),
            Dict{Any,Int}(), Dict{Any,UInt64}(), Dict{Any,Int}(),
            Random.Xoshiro(UInt64(seed)), 0, 0, 0, 0, 0,
            policy == :tinylfu ? zeros(UInt8,4,frequency_slots) : zeros(UInt8,0,0),
            0, 0, Any[], Dict{Any,Bool}(), 0, 0, ReentrantLock())
    end
end

function _cache_frequency_index(cache::ProductPlanCache,key,row::Int)
    slots=size(cache.frequency,2)
    seed=UInt(0x9e3779b97f4a7c15) ⊻ (UInt(row)*UInt(0xbf58476d1ce4e5b9))
    Int(mod(hash(key,seed),UInt(slots)))+1
end

function _cache_frequency(cache::ProductPlanCache,key)
    minimum(cache.frequency[row,_cache_frequency_index(cache,key,row)] for row in 1:4)
end

function _cache_record_frequency!(cache::ProductPlanCache,key)
    cache.policy == :tinylfu || return
    for row in 1:4
        index=_cache_frequency_index(cache,key,row)
        value=cache.frequency[row,index]
        value<typemax(UInt8) && (cache.frequency[row,index]=value+one(UInt8))
    end
    cache.frequency_samples+=1
    if cache.frequency_samples>=10*size(cache.frequency,2)
        for index in eachindex(cache.frequency)
            cache.frequency[index] >>= 1
        end
        cache.frequency_samples=0
    end
end

function _plan_cache_key(a::AbstractMultiVector, b::AbstractMultiVector,
                         operation::Symbol, max_paths::Integer)
    ga = _same_algebra(a, b)
    isdiag(metric(ga)) ||
        throw(ArgumentError("cached product currently requires a diagonal metric"))
    asupport = sort!([mask for (mask, _) in _terms(a)])
    bsupport = sort!([mask for (mask, _) in _terms(b)])
    return (dimension(ga), kind(ga), eltype(metric(ga)),
            Tuple(diag(metric(ga))),
            Tuple(basis(ga)), operation, max_paths,
            Tuple(asupport), Tuple(bsupport))
end

# `summarysize((key, plan))` recursively walks the metric, arrays, sets and
# strings on every miss. A cache estimate need not reproduce the allocator's
# exact layout, but must charge for all retained structures. Use deliberately
# generous per-entry overhead for fixed-width masks and metric coefficients;
# fall back to the exact traversal for heap-allocated scalar/mask types.
function _plan_cache_size(key,plan::ProductPlan{T,K}) where {T,K}
    if !isbitstype(T) || !isbitstype(K)
        return Base.summarysize((key,plan))
    end
    n=length(plan.diagonal)
    matrix_bytes=Base.checked_mul(Base.checked_mul(n,n),sizeof(T))
    basis_bytes=sum(96+ncodeunits(name) for name in plan.basis_names;init=0)
    support_entries=length(plan.left_masks)+length(plan.right_masks)+
        length(plan.output_masks)
    return 2048+matrix_bytes+2*n*sizeof(T)+2*basis_bytes+
        128*support_entries+(128+sizeof(T))*length(plan.paths)
end

function _cache_touch!(cache::ProductPlanCache, key; hit::Bool=true)
    cache.clock += 1
    cache.ages[key] = cache.clock
    hit && (cache.hit_counts[key] += 1)
end

function _cache_victim(cache::ProductPlanCache)
    if cache.policy == :sieve
        while true
            cache.sieve_hand==0 && (cache.sieve_hand=1)
            key=cache.sieve_order[cache.sieve_hand]
            cache.sieve_scans+=1
            if !cache.sieve_visited[key]
                return key
            end
            cache.sieve_visited[key]=false
            cache.sieve_hand=mod1(cache.sieve_hand+1,length(cache.sieve_order))
        end
    elseif cache.policy in (:lru,:tinylfu)
        oldest = first(keys(cache.plans))
        for key in keys(cache.plans)
            cache.ages[key] < cache.ages[oldest] && (oldest = key)
        end
        return oldest
    end
    # Stable order makes a fixed roulette seed reproducible across processes.
    candidates = sort!(collect(keys(cache.plans)); by=key -> cache.ages[key])
    # Inverse-frequency roulette: frequently reused plans are less likely to
    # leave. Every plan retains positive eviction probability. The draw changes
    # only retained plans, never algebraic contributions to a product.
    weights = [(1.0 + cache.hit_counts[key])^(-cache.eviction_exponent)
               for key in candidates]
    draw = rand(cache.rng) * sum(weights)
    for (key, weight) in zip(candidates, weights)
        draw -= weight
        draw < 0 && return key
    end
    return last(candidates)
end

function _cache_remove_sieve!(cache::ProductPlanCache,key)
    cache.policy == :sieve || return
    index=findfirst(isequal(key),cache.sieve_order)
    isnothing(index) && error("SIEVE victim is absent from its queue")
    deleteat!(cache.sieve_order,index)
    delete!(cache.sieve_visited,key)
    if isempty(cache.sieve_order)
        cache.sieve_hand=0
    elseif index<cache.sieve_hand
        cache.sieve_hand-=1
    elseif cache.sieve_hand>length(cache.sieve_order)
        cache.sieve_hand=1
    end
end

"""
    cached_plan!(cache, a, b; operation=:geometric, max_paths=65536)

Return a structural plan. Keys include a content snapshot of the diagonal
metric, basis names, operation, supports and path budget; input coefficients
are never cached. Construction occurs outside the lock. Retention uses a
    conservative structural estimate, not an exact process-memory guarantee.
"""
function cached_plan!(cache::ProductPlanCache,
                      a::AbstractMultiVector, b::AbstractMultiVector;
                      operation::Symbol=:geometric,
                      max_paths::Integer=1 << 16)
    key = _plan_cache_key(a, b, operation, max_paths)
    lock(cache.lock)
    try
        if haskey(cache.plans, key)
            cache.hits += 1
            _cache_touch!(cache, key)
            _cache_record_frequency!(cache,key)
            cache.policy == :sieve && (cache.sieve_visited[key]=true)
            return cache.plans[key]
        end
        cache.misses += 1
        _cache_record_frequency!(cache,key)
    finally
        unlock(cache.lock)
    end

    plan = prepare_product(a, b; operation, max_paths)
    size = _plan_cache_size(key, plan)
    size > cache.max_bytes && return plan

    lock(cache.lock)
    try
        if haskey(cache.plans, key)
            cache.hits += 1
            _cache_touch!(cache, key)
            cache.policy == :sieve && (cache.sieve_visited[key]=true)
            return cache.plans[key]
        end
        if cache.policy == :tinylfu && cache.used_bytes > cache.max_bytes - size
            victim=_cache_victim(cache)
            if _cache_frequency(cache,key) <= _cache_frequency(cache,victim)
                cache.admissions_rejected+=1
                return plan
            end
        end
        while cache.used_bytes > cache.max_bytes - size
            length(cache.plans)>=2 && (cache.multi_candidate_evictions += 1)
            victim = _cache_victim(cache)
            cache.used_bytes -= pop!(cache.sizes, victim)
            delete!(cache.plans, victim)
            delete!(cache.ages, victim)
            delete!(cache.hit_counts, victim)
            _cache_remove_sieve!(cache,victim)
            cache.evictions += 1
        end
        cache.plans[key] = plan
        cache.sizes[key] = size
        cache.used_bytes += size
        cache.hit_counts[key] = 0
        _cache_touch!(cache, key; hit=false)
        if cache.policy == :sieve
            push!(cache.sieve_order,key)
            cache.sieve_visited[key]=false
            cache.sieve_hand==0 && (cache.sieve_hand=1)
        end
        return plan
    finally
        unlock(cache.lock)
    end
end

"""Multiply using a plan cache; all input values are read fresh."""
cached_product!(cache::ProductPlanCache, a::AbstractMultiVector,
                b::AbstractMultiVector; kwargs...) =
    run_product(cached_plan!(cache, a, b; kwargs...), a, b)

function cache_stats(cache::ProductPlanCache)
    lock(cache.lock)
    try
        return (entries=length(cache.plans), estimated_bytes=cache.used_bytes,
                max_bytes=cache.max_bytes, policy=cache.policy,
                eviction_exponent=cache.eviction_exponent, hits=cache.hits,
                misses=cache.misses, evictions=cache.evictions,
                multi_candidate_evictions=cache.multi_candidate_evictions,
                admissions_rejected=cache.admissions_rejected,
                frequency_metadata_bytes=length(cache.frequency),
                sieve_scans=cache.sieve_scans)
    finally
        unlock(cache.lock)
    end
end

function Base.empty!(cache::ProductPlanCache)
    lock(cache.lock)
    try
        empty!(cache.plans)
        empty!(cache.sizes)
        empty!(cache.ages)
        empty!(cache.hit_counts)
        fill!(cache.frequency,0)
        empty!(cache.sieve_order)
        empty!(cache.sieve_visited)
        cache.used_bytes = 0
        cache.clock = 0
        cache.hits = 0
        cache.misses = 0
        cache.evictions = 0
        cache.multi_candidate_evictions = 0
        cache.frequency_samples = 0
        cache.admissions_rejected = 0
        cache.sieve_hand = 0
        cache.sieve_scans = 0
        return cache
    finally
        unlock(cache.lock)
    end
end
