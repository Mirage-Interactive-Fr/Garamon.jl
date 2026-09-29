using Garamon
using LinearAlgebra
using SHA

const LONG_HORIZON_STRATEGIES = (:direct, :prepared, :generated, :recursive, :join3, :workspace)

function long_source_hash()
    source = joinpath(@__DIR__, "..", "src")
    files = sort!(filter(path -> endswith(path, ".jl"), readdir(source; join=true)))
    bytes2hex(sha256(join((basename(path) * "\n" * read(path, String) for path in files), "\n")))
end

function long_rss()
    isfile("/proc/self/status") || return -1
    match_result = match(r"(?m)^VmRSS:\s+(\d+)\s+kB", read("/proc/self/status", String))
    match_result === nothing ? -1 : 1024parse(Int, match_result.captures[1])
end

function long_peak_rss()
    isfile("/proc/self/status") || return -1
    match_result = match(r"(?m)^VmHWM:\s+(\d+)\s+kB", read("/proc/self/status", String))
    match_result === nothing ? -1 : 1024parse(Int, match_result.captures[1])
end

# Independent exact exterior-basis oracle: explicit inversion count, no Garamon
# multiplication, sign, indexing, or coefficient-selection helper.
function long_indices(mask::Integer)
    indices = Int[]
    index = 1
    while !iszero(mask)
        isodd(mask) && push!(indices, index)
        mask >>= 1
        index += 1
    end
    indices
end

function long_reference_product(left, right)
    K = keytype(left)
    result = Dict{K,Int64}()
    for (a, avalue) in left, (b, bvalue) in right
        parity = sum((i > j for i in long_indices(a) for j in long_indices(b)); init=0)
        value = isodd(parity) ? -avalue*bvalue : avalue*bvalue
        result[xor(a, b)] = get(result, xor(a, b), Int64(0)) + value
    end
    filter!(pair -> !iszero(last(pair)), result)
    result
end

long_value(t, bank, operand, slot) = Int64(1 + mod(t + 2bank + 3operand + slot, 3))

function long_bank(ga, bank)
    n = dimension(ga)
    K = n <= 64 ? UInt64 : BigInt
    # Directions are dispersed across the ambient algebra, including bits above
    # UInt64. Operand subspaces are disjoint: every a*b coefficient has exactly
    # one contribution, so coefficient changes cannot invalidate its support.
    directions = sort!([1 + mod(7bank, n÷4),
                        1 + n÷3 + mod(bank, 7),
                        1 + 2n÷3 + mod(bank, 7), n])
    length(unique(directions)) == 4 || error("benchmark directions must be distinct")
    bits = [one(K) << (i-1) for i in directions]
    amasks = sort!(K[0, bits[1], bits[3], bits[1] | bits[3]])
    bmasks = sort!(K[0, bits[2], bits[4], bits[2] | bits[4]])
    abmasks = sort!(K[xor(a, b) for a in amasks for b in bmasks])
    length(unique(abmasks)) == 16 || error("intermediate support is not unique")
    # Vary final contraction partners as well as absolute coordinate support.
    cmasks = sort!(unique(K[abmasks[1], abmasks[2 + mod(bank, 4)],
                            abmasks[7 + mod(bank, 4)], abmasks[16]]))
    masks = (amasks, bmasks, cmasks)
    operands = ntuple(3) do operand
        multivector(ga, Dict(mask => Float64(long_value(0, bank, operand, slot))
            for (slot, mask) in enumerate(masks[operand])); storage=:sparse)
    end
    prototype = multivector(ga, Dict(mask => 1.0 for mask in abmasks); storage=:sparse)
    expected = Float64[]
    for t in 1:3
        refs = ntuple(3) do operand
            Dict(mask => long_value(t, bank, operand, slot)
                 for (slot, mask) in enumerate(masks[operand]))
        end
        exact = long_reference_product(long_reference_product(refs[1], refs[2]), refs[3])
        push!(expected, Float64(get(exact, zero(K), Int64(0))))
    end
    # Every absolute partial sum is <= 4^3*3^3, hence exact in Float64 and Int64.
    return (; operands, masks, prototype, expected)
end

function long_state(n::Int, trace::Symbol, horizon::Int)
    n in (66, 128) || throw(ArgumentError("admitted dimensions: 66, 128"))
    trace in (:stable, :varying, :phases) || throw(ArgumentError("unknown trace"))
    horizon in (1, 32, 1024, 10000) || throw(ArgumentError("unbounded horizon"))
    bank_count = trace == :stable ? 1 : 8
    banks = [long_bank(algebra(n, :ega), bank) for bank in 1:bank_count]
    sequence = if trace == :stable
        ones(Int, horizon)
    elseif trace == :varying
        [1 + mod(t-1, bank_count) for t in 1:horizon]
    else
        # Eight consecutive support phases; no scaling of a one-step timing.
        [min(bank_count, 1 + fld((t-1)*bank_count, horizon)) for t in 1:horizon]
    end
    expected = [banks[sequence[t]].expected[1 + mod(t-1, 3)] for t in 1:horizon]
    return (; n, trace, horizon, banks, sequence, expected)
end

function long_update!(bank, t, bank_id)
    for operand in 1:3
        values = bank.operands[operand].values
        for (slot, mask) in enumerate(bank.masks[operand])
            values[mask] = Float64(long_value(t, bank_id, operand, slot))
        end
    end
    nothing
end

function long_prepare(bank, strategy)
    a, b, c = bank.operands
    if strategy in (:recursive, :join3)
        return prepare_expression(@ga (a*b)*c)
    elseif strategy in (:prepared, :generated, :workspace)
        left = prepare_product(a, b; max_paths=256)
        right = prepare_product(bank.prototype, c; max_paths=256)
        strategy == :workspace && return (
            ProductWorkspace(left, a, b; max_bytes=1 << 20),
            ProductWorkspace(right, bank.prototype, c; max_bytes=1 << 20))
        return strategy == :prepared ? (left, right) :
            (generate_product(left; max_paths=256), generate_product(right; max_paths=256))
    elseif strategy == :direct
        return nothing
    end
    throw(ArgumentError("unknown long-horizon strategy"))
end

function long_step(bank, artifact, strategy)
    a, b, c = bank.operands
    if strategy == :direct
        return scalarpart((a*b)*c)
    elseif strategy == :prepared
        ab = run_product(artifact[1], a, b)
        return scalarpart(run_product(artifact[2], ab, c))
    elseif strategy == :generated
        ab = run_generated_product(artifact[1], a, b)
        return scalarpart(run_generated_product(artifact[2], ab, c))
    elseif strategy == :workspace
        ab = run_product!(artifact[1], a, b)
        return scalarpart(run_product!(artifact[2], ab, c))
    elseif strategy in (:recursive, :join3)
        return evaluate(artifact; output=Int[], strategy,
                        max_support=4096, max_pairs=1 << 16)
    end
    throw(ArgumentError("unknown strategy"))
end

function long_trace(state, strategy)
    strategy in LONG_HORIZON_STRATEGIES || throw(ArgumentError("strategy not admitted"))
    artifacts = Vector{Any}(nothing, length(state.banks))
    prepared = falses(length(state.banks))
    values = Vector{Float64}(undef, state.horizon)
    builds = 0
    for t in 1:state.horizon
        bank_id = state.sequence[t]
        bank = state.banks[bank_id]
        long_update!(bank, t, bank_id)
        if !prepared[bank_id]
            artifacts[bank_id] = long_prepare(bank, strategy)
            prepared[bank_id] = true
            builds += strategy == :direct ? 0 : 1
        end
        values[t] = long_step(bank, artifacts[bank_id], strategy)
    end
    return (; values, artifacts, builds)
end

function long_observe(f, args...)
    @nospecialize f args
    @timed Base.invokelatest(f, args...)
end

function long_setup_with_cold(n, trace, horizon, strategy, destination)
    BLAS.set_num_threads(1)
    fingerprint = long_source_hash()
    state = long_state(n, trace, horizon)
    # No Garamon product or expression evaluator has run for the fixture/oracle.
    long_observe(identity, nothing)
    GC.gc()
    rss_before, peak_before, jit_before = long_rss(), long_peak_rss(), Base.jit_total_bytes()
    measured = long_observe(long_trace, state, strategy)
    jit_added = Base.jit_total_bytes() - jit_before
    rss_after, peak_after = long_rss(), long_peak_rss()
    measured.value.values == state.expected || error("cold long trace failed exact oracle")
    fingerprint == long_source_hash() || error("source changed during cold trace")
    GC.gc()
    row = (dimension=n, trace=String(trace), horizon=horizon, strategy=String(strategy),
        cold_ms=1000measured.time, compile_ms=1000measured.compile_time,
        recompile_ms=1000measured.recompile_time, cold_allocated_bytes=measured.bytes,
        cold_gc_ms=1000measured.gctime, jit_added_bytes=jit_added,
        rss_before_bytes=rss_before, rss_after_bytes=rss_after,
        rss_after_gc_bytes=long_rss(), peak_rss_before_bytes=peak_before,
        peak_rss_after_bytes=peak_after,
        retained_state_result_bytes=Base.summarysize((state, measured.value)),
        artifacts_built=measured.value.builds, source_sha256=fingerprint)
    # PerfChecker can create more than one worker for qualification/measurement.
    # Each feature's first cold observation is preserved, never overwritten.
    if !isfile(destination)
        open(destination, "w") do io
            println(io, join(keys(row), ','))
            println(io, join(values(row), ','))
        end
    end
    return state
end

function long_oracle(state, strategy)
    result = long_trace(state, strategy)
    result.values == state.expected &&
        result.builds == (strategy == :direct ? 0 : length(unique(state.sequence)))
end
