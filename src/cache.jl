"""A byte-bounded, thread-safe cache of structural product plans."""
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
    lock::ReentrantLock
    function ProductPlanCache(; max_bytes::Integer=1 << 20,
                              policy::Symbol=:lru,
                              eviction_exponent::Real=1.0,
                              seed::Integer=0x676172616d6f6e)
        0 <= max_bytes <= typemax(Int) ||
            throw(ArgumentError("cache byte budget must be a nonnegative Int"))
        policy in (:lru, :roulette) ||
            throw(ArgumentError("cache policy must be :lru or :roulette"))
        isfinite(eviction_exponent) && 0 <= eviction_exponent <= 8 ||
            throw(ArgumentError("cache eviction exponent must be finite in [0,8]"))
        0 <= seed <= typemax(UInt64) ||
            throw(ArgumentError("cache seed must fit UInt64"))
        new(Int(max_bytes), policy, Float64(eviction_exponent), 0,
            Dict{Any,ProductPlan}(),
            Dict{Any,Int}(), Dict{Any,UInt64}(), Dict{Any,Int}(),
            Random.Xoshiro(UInt64(seed)), 0, 0, 0, 0, 0, ReentrantLock())
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

function _cache_touch!(cache::ProductPlanCache, key; hit::Bool=true)
    cache.clock += 1
    cache.ages[key] = cache.clock
    hit && (cache.hit_counts[key] += 1)
end

function _cache_victim(cache::ProductPlanCache)
    if cache.policy == :lru
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

"""
    cached_plan!(cache, a, b; operation=:geometric, max_paths=65536)

Return a structural plan. Keys include a content snapshot of the diagonal
metric, basis names, operation, supports and path budget; input coefficients
are never cached. Construction occurs outside the lock. Retention uses a
conservative `summarysize` estimate, not an exact process-memory guarantee.
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
            return cache.plans[key]
        end
        cache.misses += 1
    finally
        unlock(cache.lock)
    end

    plan = prepare_product(a, b; operation, max_paths)
    size = Base.summarysize((key, plan))
    size > cache.max_bytes && return plan

    lock(cache.lock)
    try
        if haskey(cache.plans, key)
            cache.hits += 1
            _cache_touch!(cache, key)
            return cache.plans[key]
        end
        while cache.used_bytes > cache.max_bytes - size
            length(cache.plans)>=2 && (cache.multi_candidate_evictions += 1)
            victim = _cache_victim(cache)
            cache.used_bytes -= pop!(cache.sizes, victim)
            delete!(cache.plans, victim)
            delete!(cache.ages, victim)
            delete!(cache.hit_counts, victim)
            cache.evictions += 1
        end
        cache.plans[key] = plan
        cache.sizes[key] = size
        cache.used_bytes += size
        cache.hit_counts[key] = 0
        _cache_touch!(cache, key; hit=false)
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
                multi_candidate_evictions=cache.multi_candidate_evictions)
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
        cache.used_bytes = 0
        cache.clock = 0
        cache.hits = 0
        cache.misses = 0
        cache.evictions = 0
        cache.multi_candidate_evictions = 0
        return cache
    finally
        unlock(cache.lock)
    end
end
