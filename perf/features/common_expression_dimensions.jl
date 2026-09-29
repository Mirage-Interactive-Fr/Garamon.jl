using Garamon

# Work bounds are checked before constructing an algebra or an exact oracle.
# seconds is BenchmarkTools' sampling target, not a preemptive timeout.
const EXPRESSION_DIMENSION_FAMILIES = (:cap12, :local12, :mixed12, :full_grade2, :dense)
function expression_dimension_limits(shape::Symbol)
    shape in EXPRESSION_DIMENSION_FAMILIES || throw(ArgumentError("unknown family $shape"))
    large = shape == :full_grade2
    return (; max_dimension=shape == :cap12 ? 256 : shape == :dense ? 5 : large ? 11 : 70,
            max_metric_entries=shape == :cap12 ? 65_536 : 4_900,
            max_setup_bytes=256 << 20,
            max_terms=large ? 64 : shape == :dense ? 32 : 13,
            max_oracle_pairs=large ? 270_336 : shape == :dense ? 33_792 : 2_366,
            max_support=large ? 4096 : shape == :dense ? 32 : 2197,
            max_pairs=large ? 1 << 20 : 1 << 18,
            max_alloc_bytes=large ? 128 << 20 : 64 << 20,
            samples=12, seconds=0.05)
end

function expression_dimension_support(n::Int, shape::Symbol)
    shape == :dense && return 1 << n
    shape == :full_grade2 && return binomial(n, 2) + 1
    shape in (:cap12, :local12) && return min(12, binomial(n, 2)) + 1
    shape == :mixed12 && return min(12, n + binomial(n, 2) + max(0, n - 2)) + 1
    throw(ArgumentError("unknown family $shape"))
end

function expression_dimension_admission(n::Int, shape::Symbol)
    n >= 2 || return "dimension_below_two"
    limits = expression_dimension_limits(shape)
    n^2 <= limits.max_metric_entries || return "family_metric_entry_limit"
    n <= limits.max_dimension || return "family_dimension_limit"
    s = expression_dimension_support(n, shape)
    s <= limits.max_terms || return "family_term_limit"
    s^2 + s^3 <= limits.max_oracle_pairs || return "family_oracle_pair_limit"
    return "admitted"
end

# Independent oracle: explicit basis-index inversions, integer arithmetic,
# no Garamon product, sign, coefficient, grade, or indexing function.
function expression_reference_indices(mask::Integer)
    result = Int[]
    i = 1
    while !iszero(mask)
        isodd(mask) && push!(result, i)
        mask >>= 1
        i += 1
    end
    return result
end

function expression_reference_product(left::Dict{K,Int64}, right::Dict{K,Int64}) where K
    result = Dict{K,Int64}()
    left_indices = Dict(k => expression_reference_indices(k) for k in keys(left))
    right_indices = Dict(k => expression_reference_indices(k) for k in keys(right))
    for (a, x) in left, (b, y) in right
        inversions = sum((i > j for i in left_indices[a] for j in right_indices[b]); init=0)
        coefficient = isodd(inversions) ? -x*y : x*y
        key = xor(a, b)
        result[key] = get(result, key, Int64(0)) + coefficient
    end
    filter!(pair -> !iszero(last(pair)), result)
    return result
end

function expression_dimension_state(n::Int, shape::Symbol, request::Symbol=:scalar)
    admission = expression_dimension_admission(n, shape)
    admission == "admitted" || throw(ArgumentError(admission))
    request in (:scalar, :four) || throw(ArgumentError("unknown request $request"))
    ga = algebra(n, :ega)
    K = n <= 64 ? UInt64 : BigInt
    blade(indices...) = foldl(|, (one(K) << (i-1) for i in indices); init=zero(K))
    grade2 = K[blade(i, j) for i in 1:n-1 for j in i+1:n]
    candidates = shape == :mixed12 ?
        vcat(K[blade(i) for i in 1:n], grade2,
             K[blade(i, i+1, i+2) for i in 1:n-2]) : grade2
    masks = if shape == :dense
        K.(1:(1 << n)-1)
    elseif shape == :full_grade2
        grade2
    elseif shape == :local12
        first(grade2, min(12, length(grade2)))
    else
        selected = K[]
        # Dispersed support touches the high bits, including the UInt64 boundary.
        for index in round.(Int, range(1, length(candidates); length=min(12, length(candidates))))
            candidates[index] in selected || push!(selected, candidates[index])
        end
        for mask in candidates
            length(selected) >= min(12, length(candidates)) && break
            mask in selected || push!(selected, mask)
        end
        selected
    end
    references = ntuple(3) do operand
        values = Dict{K,Int64}(zero(K) => Int64(operand))
        for (k, mask) in enumerate(masks)
            v = mod((operand + 1)*k + operand, 7) - 3
            values[mask] = iszero(v) ? 1 : v
        end
        values
    end
    # |coefficient| and every partial sum <= 27*s^3, hence exact in Float64.
    # The family caps also ensure this is far below typemax(Int64).
    s = length(masks) + 1
    @assert 27 * s^3 < 2^53
    a, b, c = map(references) do values
        multivector(ga, Dict{K,Float64}(k => Float64(v) for (k, v) in values); storage=:sparse)
    end
    plan = prepare_expression(@ga (a * b) * c)
    ab = expression_reference_product(references[1], references[2])
    reference = expression_reference_product(ab, references[3])
    targets = request == :scalar ? K[0] : unique(K[0, blade(1), blade(1, n), blade(n)])
    requests = expression_reference_indices.(targets)
    expected = Float64[get(reference, mask, Int64(0)) for mask in targets]
    return (; plan, expected, targets, requests, support=s,
             request_count=length(targets), reference_support=length(reference),
             oracle_pairs=s^2 + length(ab)*s, limits=expression_dimension_limits(shape))
end

function expression_dimension_workload(state, strategy::Symbol)
    if strategy == :full
        result = evaluate(state.plan)
        return [coefficient_mask(result, mask) for mask in state.targets]
    end
    return evaluate(state.plan; outputs=state.requests, strategy,
                    max_support=state.limits.max_support, max_pairs=state.limits.max_pairs)
end

expression_dimension_oracle(state, strategy::Symbol) =
    expression_dimension_workload(state, strategy) == state.expected
