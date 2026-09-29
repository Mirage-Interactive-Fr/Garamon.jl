using Garamon

function make_state(trace::Symbol)
    ga = algebra(12, :ega)
    left = multivector(ga, Dict(UInt64(i) => Float64(i) for i in 1:16);
                       storage=:sparse)
    rights = [multivector(ga,
        Dict(UInt64(32j + i) => Float64(i - j) for i in 1:16 if i != j);
        storage=:sparse) for j in 1:8]
    sequence = if trace == :hot
        vcat(repeat([1, 2, 1, 2, 1, 2, 3, 1, 2, 4], 4)...)
    elseif trace == :scan
        vcat(repeat([1, 2, 1, 2, 3, 4, 5, 6, 7, 8], 4)...)
    elseif trace == :phase
        vcat(fill(1, 10), fill(2, 10), fill(3, 10), fill(4, 10))
    elseif trace == :shift
        vcat(repeat([1, 2], 20), repeat([3, 4], 20))
    else
        error("unknown trace")
    end
    probe = ProductPlanCache(max_bytes=typemax(Int))
    largest = 0
    for right in rights
        empty!(probe)
        cached_plan!(probe, left, right)
        largest = max(largest, cache_stats(probe).estimated_bytes)
    end
    budget = 2largest
    expected = [left * rights[i] for i in sequence]
    return (; left, rights, sequence, budget, expected)
end

function run_trace(state, policy::Symbol)
    cache = policy == :direct ? nothing :
        ProductPlanCache(max_bytes=state.budget, policy=policy, seed=42)
    output = Vector{AbstractMultiVector}(undef, length(state.sequence))
    for (k, index) in enumerate(state.sequence)
        output[k] = cache === nothing ?
            state.left * state.rights[index] :
            cached_product!(cache, state.left, state.rights[index])
    end
    return (; output, stats=cache === nothing ? nothing : cache_stats(cache))
end

function trace_oracle(state, policy::Symbol)
    result = run_trace(state, policy)
    return result.output == state.expected &&
        (result.stats === nothing ||
         (result.stats.estimated_bytes <= result.stats.max_bytes &&
          result.stats.entries <= 2))
end
