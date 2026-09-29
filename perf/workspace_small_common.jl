using Garamon

const WORKSPACE_SMALL_FAMILIES = (:cap12, :full_grade2, :dense)
const WORKSPACE_SMALL_HORIZONS = (1, 32, 1024)
const WORKSPACE_SMALL_STRATEGIES = (:direct, :prepared, :workspace, :values)
const WORKSPACE_SMALL_LIMITS = (
    max_dimension=12, max_paths=65_536, max_batch_paths=262_144,
    max_workspace_bytes=64 << 20, max_fixture_alloc_bytes=256 << 20,
    max_batch_alloc_bytes=256 << 20, samples=12, seconds=0.1)

function workspace_small_support(n::Int, family::Symbol)
    family == :dense && return 1 << n
    family == :full_grade2 && return binomial(n, 2) + 1
    family == :cap12 && return min(12, binomial(n, 2)) + 1
    throw(ArgumentError("unknown workspace family $family"))
end

function workspace_small_admission(n::Int, family::Symbol, horizon::Int=1)
    2 <= n <= WORKSPACE_SMALL_LIMITS.max_dimension || return "dimension_limit"
    horizon >= 1 || return "invalid_horizon"
    paths = workspace_small_support(n, family)^2
    paths <= WORKSPACE_SMALL_LIMITS.max_paths || return "plan_path_limit"
    paths * horizon <= WORKSPACE_SMALL_LIMITS.max_batch_paths || return "batch_path_limit"
    return "admitted"
end

# Independent integer reference: explicit index inversions, no Garamon product,
# sign, coefficient, or index helper. Euclidean basis blade multiplication.
function workspace_small_indices(mask::UInt64)
    indices = Int[]
    index = 1
    while !iszero(mask)
        isodd(mask) && push!(indices, index)
        mask >>= 1
        index += 1
    end
    return indices
end

function workspace_small_reference(a::Dict{UInt64,Int64}, b::Dict{UInt64,Int64})
    ai = Dict(mask => workspace_small_indices(mask) for mask in keys(a))
    bi = Dict(mask => workspace_small_indices(mask) for mask in keys(b))
    result = Dict{UInt64,Int64}()
    for (amask, x) in a, (bmask, y) in b
        inversions = sum((i > j for i in ai[amask] for j in bi[bmask]); init=0)
        contribution = isodd(inversions) ? -x*y : x*y
        mask = xor(amask, bmask)
        result[mask] = get(result, mask, Int64(0)) + contribution
    end
    filter!(pair -> !iszero(last(pair)), result)
    return result
end

function workspace_small_fixture(n::Int, family::Symbol)
    workspace_small_admission(n, family) == "admitted" || error("unadmitted fixture")
    ga = algebra(n, :ega)
    blade(i, j) = (UInt64(1) << (i-1)) | (UInt64(1) << (j-1))
    masks = if family == :dense
        UInt64.(0:(1 << n)-1)
    else
        candidates = UInt64[blade(i, j) for i in 1:n-1 for j in i+1:n]
        selected = family == :full_grade2 ? candidates :
            candidates[unique(round.(Int, range(1, length(candidates);
                                                   length=min(12, length(candidates)))))]
        vcat(UInt64[0], selected)
    end
    references = ntuple(4) do variant
        ntuple(2) do operand
            Dict{UInt64,Int64}(mask => begin
                value = mod((2operand + 1)*k + 3variant + operand, 7) - 3
                Int64(iszero(value) ? 1 : value)
            end for (k, mask) in enumerate(masks))
        end
    end
    inputs = map(references) do pair
        map(pair) do coefficients
            multivector(ga, Dict(k => Float64(v) for (k, v) in coefficients);
                        storage=family == :dense ? :dense : :sparse)
        end
    end
    expected = map(pair -> workspace_small_reference(pair...), references)
    checksums = map(reference -> Float64(sum(values(reference); init=Int64(0))), expected)
    # This covers every partial sum in a batch, not just its final coefficient.
    @assert 9 * WORKSPACE_SMALL_LIMITS.max_batch_paths < 2^53
    return (; n, family, inputs, expected, checksums, support=length(masks))
end

workspace_small_artifact(state, ::Val{:direct}) = nothing
workspace_small_artifact(state, ::Val{:prepared}) =
    prepare_product(state.inputs[1]...; max_paths=WORKSPACE_SMALL_LIMITS.max_paths)
function workspace_small_artifact(state, ::Union{Val{:workspace},Val{:values}})
    plan = workspace_small_artifact(state, Val(:prepared))
    return ProductWorkspace(plan, state.inputs[1]...;
                            max_bytes=WORKSPACE_SMALL_LIMITS.max_workspace_bytes)
end

workspace_small_step(state, ::Val{:direct}, _, i) = state.inputs[i][1] * state.inputs[i][2]
workspace_small_step(state, ::Val{:prepared}, plan, i) = run_product(plan, state.inputs[i]...)
workspace_small_step(state, ::Val{:workspace}, workspace, i) = run_product!(workspace, state.inputs[i]...)
workspace_small_step(state, ::Val{:values}, workspace, i) = run_product_values!(workspace, state.inputs[i]...)
workspace_small_consume(result::Garamon.SparseMultiVector) = sum(values(result.values); init=0.0)
workspace_small_consume(result::Garamon.DenseMultiVector) = sum(result.values; init=0.0)
workspace_small_consume(result::AbstractVector) = sum(result; init=0.0)

# Function barrier: preparation can return an imprecisely inferred plan/workspace
# type. Dispatch once on the resulting artifact, not once per iteration.
@noinline function workspace_small_loop(state, strategy::Val, horizon::Int, artifact)
    checksum = 0.0
    for iteration in 1:horizon
        # Identical support, different coefficients; exercise buffer refresh.
        index = mod1(iteration, length(state.inputs))
        checksum += workspace_small_consume(workspace_small_step(state, strategy, artifact, index))
    end
    return checksum
end

workspace_small_batch(state, strategy::Val, horizon::Int, ::Val{:steady}, artifact) =
    workspace_small_loop(state, strategy, horizon, artifact)
workspace_small_batch(state, strategy::Val, horizon::Int, ::Val{:end_to_end}, _) =
    workspace_small_loop(state, strategy, horizon, workspace_small_artifact(state, strategy))

function workspace_small_expected_checksum(state, horizon::Int)
    cycles, tail = divrem(horizon, length(state.inputs))
    return cycles * sum(state.checksums) + sum(state.checksums[1:tail]; init=0.0)
end

function workspace_small_oracle(state, strategy::Val{S}, artifact) where S
    for i in eachindex(state.inputs)
        result = workspace_small_step(state, strategy, artifact, i)
        if S == :values
            expected = [Float64(get(state.expected[i], mask, 0)) for mask in artifact.plan.output_masks]
            result == expected || return false
        elseif result isa Garamon.DenseMultiVector
            expected = [Float64(get(state.expected[i], UInt64(mask), 0))
                        for mask in 0:length(result.values)-1]
            result.values == expected || return false
        else
            result.values == state.expected[i] || return false
        end
    end
    return true
end

workspace_small_prepare(state, _, ::Val{:plan_build}) = workspace_small_artifact(state, Val(:prepared))
workspace_small_prepare(state, plan, ::Val{:workspace_build}) =
    ProductWorkspace(plan, state.inputs[1]...; max_bytes=WORKSPACE_SMALL_LIMITS.max_workspace_bytes)
workspace_small_prepare(state, _, ::Val{:plan_workspace_build}) =
    workspace_small_artifact(state, Val(:workspace))
