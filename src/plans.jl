"""A reusable, bounded structure plan for a diagonal-metric product."""
struct ProductPlan{T,K<:Integer,A<:GeometricAlgebra}
    algebra::A
    operation::Symbol
    diagonal::Vector{T}
    basis_names::Vector{String}
    left_support::Set{K}
    right_support::Set{K}
    left_masks::Vector{K}
    right_masks::Vector{K}
    output_masks::Vector{K}
    paths::Vector{Tuple{Int,Int,Int,T}}
end

"""
    prepare_product(a, b; operation=:geometric, max_paths=65536)

Prepare only the nonzero blade-pair paths for repeated products with identical
supports. The plan contains no input coefficients; `run_product` checks the
supports before use. The construction budget includes paths that later vanish.
"""
function prepare_product(a::AbstractMultiVector, b::AbstractMultiVector;
                         operation::Symbol=:geometric,
                         max_paths::Integer=1 << 16)
    ga = _same_algebra(a, b)
    isdiag(metric(ga)) ||
        throw(ArgumentError("prepared product currently requires a diagonal metric"))
    operation in _PRODUCT_OPERATIONS ||
        throw(ArgumentError("unsupported product operation"))
    max_paths >= 0 || throw(ArgumentError("max_paths must be nonnegative"))
    K = _masktype(ga)
    T = eltype(metric(ga))
    asupport = Set{K}(convert(K, mask) for (mask, _) in _terms(a))
    bsupport = Set{K}(convert(K, mask) for (mask, _) in _terms(b))
    big(length(asupport)) * length(bsupport) <= max_paths ||
        throw(ArgumentError("product plan exceeds max_paths"))
    left_masks = sort!(collect(asupport))
    right_masks = sort!(collect(bsupport))
    output_masks = K[]
    output_index = Dict{K,Int}()
    paths = Tuple{Int,Int,Int,T}[]
    for (ai, amask) in enumerate(left_masks),
        (bi, bmask) in enumerate(right_masks)
        ra, rb = _blade_grade(amask), _blade_grade(bmask)
        if operation == :wedge
            iszero(amask & bmask) || continue
        else
            selected_grade = _selected_pair_grade(operation, ra, rb)
            selected_grade === nothing && continue
            selected_grade == -1 ||
                _blade_grade(amask ⊻ bmask) == selected_grade || continue
        end
        factor = convert(T, _shuffle_sign(ga, amask, bmask))
        if operation != :wedge
            for i in _indices(ga, amask & bmask)
                factor *= metric(ga)[i, i]
            end
        end
        if !iszero(factor)
            output = amask ⊻ bmask
            oi = get!(output_index, output) do
                push!(output_masks, output)
                length(output_masks)
            end
            push!(paths, (ai, bi, oi, factor))
        end
    end
    return ProductPlan(ga, operation, collect(diag(metric(ga))), copy(basis(ga)),
                       asupport, bsupport, left_masks, right_masks,
                       output_masks, paths)
end

function _validate_plan_algebra(plan::ProductPlan, ga::GeometricAlgebra)
    valid = dimension(ga) == length(plan.diagonal) && isdiag(metric(ga)) &&
            basis(ga) == plan.basis_names &&
            kind(ga) == kind(plan.algebra)
    if valid
        for i in eachindex(plan.diagonal)
            if metric(ga)[i, i] != plan.diagonal[i]
                valid = false
                break
            end
        end
    end
    valid ||
        throw(ArgumentError("plan belongs to a different algebra or basis"))
    return ga
end

function _matches_plan_support(mv::AbstractMultiVector, support::Set;
                               allow_subsets::Bool=false)
    count = 0
    for (mask, _) in _terms(mv)
        mask in support || return false
        count += 1
    end
    return allow_subsets || count == length(support)
end

function _validate_plan_inputs(plan::ProductPlan, a::AbstractMultiVector,
                               b::AbstractMultiVector;
                               allow_subsets::Bool=false)
    ga = _validate_plan_algebra(plan, _same_algebra(a, b))
    _matches_plan_support(a, plan.left_support; allow_subsets) &&
        _matches_plan_support(b, plan.right_support; allow_subsets) ||
        throw(ArgumentError("input support differs from the prepared product"))
    return ga
end

"""Execute a prepared product; exact supports are required unless `allow_subsets=true`."""
function run_product(plan::ProductPlan{T,K}, a::AbstractMultiVector,
                     b::AbstractMultiVector;
                     allow_subsets::Bool=false) where {T,K}
    ga = _validate_plan_inputs(plan, a, b; allow_subsets)
    S = promote_type(T, eltype(a), eltype(b))
    if length(plan.paths) < length(plan.left_masks) + length(plan.right_masks)
        result = SparseMultiVector(ga, Dict{K,S}())
        for (ai, bi, oi, factor) in plan.paths
            _accumulate!(result, plan.output_masks[oi],
                         factor * coefficient_mask(a, plan.left_masks[ai]) *
                         coefficient_mask(b, plan.right_masks[bi]))
        end
        return result
    end
    left_values = S[coefficient_mask(a, mask) for mask in plan.left_masks]
    right_values = S[coefficient_mask(b, mask) for mask in plan.right_masks]
    output_values = zeros(S, length(plan.output_masks))
    for (ai, bi, oi, factor) in plan.paths
        output_values[oi] += factor * left_values[ai] * right_values[bi]
    end
    return SparseMultiVector(ga, Dict{K,S}(mask => value
        for (mask, value) in zip(plan.output_masks, output_values)
        if !iszero(value)))
end

function _grade_plan_masks(ga::GeometricAlgebra, grade::Integer,
                           max_slots::Integer)
    n = dimension(ga)
    0 <= grade <= n || throw(ArgumentError("grade is out of range"))
    binomial(big(n), grade) <= max_slots ||
        throw(ArgumentError("grade plan exceeds max_slots"))
    K = _masktype(ga)
    masks = K[]
    function visit(start::Int, remaining::Int, mask)
        if iszero(remaining)
            push!(masks, mask)
            return
        end
        for i in start:(n - remaining + 1)
            visit(i + 1, remaining - 1,
                  mask | (one(K) << (i - 1)))
        end
    end
    visit(1, Int(grade), zero(K))
    return masks
end

"""
    prepare_grade_product(ga, left_grade, right_grade;
                          operation=:geometric, max_paths=65536,
                          max_slots=65536)

Prepare all paths between two complete homogeneous grades in a diagonal
metric. The path and grade-slot budgets are checked before enumeration. The
result accepts any operands whose supports are subsets of those grades via
`run_grade_product`, including changing zero positions between calls.
"""
function prepare_grade_product(ga::GeometricAlgebra, left_grade::Integer,
                               right_grade::Integer;
                               operation::Symbol=:geometric,
                               max_paths::Integer=1 << 16,
                               max_slots::Integer=1 << 16)
    max_paths >= 0 && max_slots >= 1 ||
        throw(ArgumentError("grade plan budgets must be positive"))
    n = dimension(ga)
    0 <= left_grade <= n && 0 <= right_grade <= n ||
        throw(ArgumentError("grade is out of range"))
    left_count = binomial(big(n), left_grade)
    right_count = binomial(big(n), right_grade)
    left_count <= max_slots && right_count <= max_slots ||
        throw(ArgumentError("grade plan exceeds max_slots"))
    left_count * right_count <= max_paths ||
        throw(ArgumentError("grade product exceeds max_paths"))
    K = _masktype(ga)
    T = eltype(metric(ga))
    left_masks = _grade_plan_masks(ga, left_grade, max_slots)
    right_masks = _grade_plan_masks(ga, right_grade, max_slots)
    left = SparseMultiVector(ga, Dict{K,T}(mask => one(T)
                                           for mask in left_masks))
    right = SparseMultiVector(ga, Dict{K,T}(mask => one(T)
                                            for mask in right_masks))
    return prepare_product(left, right; operation, max_paths)
end

"""Execute a full-grade plan on homogeneous operand subsets."""
run_grade_product(plan::ProductPlan, a::AbstractMultiVector,
                  b::AbstractMultiVector) =
    run_product(plan, a, b; allow_subsets=true)

"""Run a prepared product over a homogeneous batch of input pairs."""
function batch_product(plan::ProductPlan, left::AbstractVector,
                       right::AbstractVector)
    length(left) == length(right) || throw(DimensionMismatch("batch sizes differ"))
    return [run_product(plan, left[i], right[i]) for i in eachindex(left, right)]
end

"""A validated snapshot of repeated product operands in contiguous matrices."""
struct PackedProductBatch{T,P<:ProductPlan}
    plan::P
    left_values::Matrix{T}
    right_values::Matrix{T}
end

"""
    pack_product_batch(plan, left, right)

Check the algebra and support of every pair once, then pack blade coefficients
as rows by mask and columns by batch item. The returned values are snapshots.
"""
function pack_product_batch(plan::ProductPlan, left::AbstractVector,
                            right::AbstractVector)
    length(left) == length(right) || throw(DimensionMismatch("batch sizes differ"))
    S = eltype(plan.diagonal)
    for i in eachindex(left, right)
        left[i] isa AbstractMultiVector && right[i] isa AbstractMultiVector ||
            throw(ArgumentError("batch entries must be multivectors"))
        S = promote_type(S, eltype(left[i]), eltype(right[i]))
    end
    return _pack_product_batch(plan, left, right, S)
end

function _pack_product_batch(plan::ProductPlan, left::AbstractVector,
                             right::AbstractVector, ::Type{S}) where S
    n = length(left)
    left_values = Matrix{S}(undef, length(plan.left_masks), n)
    right_values = Matrix{S}(undef, length(plan.right_masks), n)
    seen_left = IdDict{Any,Int}()
    seen_right = IdDict{Any,Int}()
    # A homogeneous batch usually reuses the plan's exact algebra object.
    # Validate its mutable metric/basis snapshot once at entry and once at
    # return, rather than scanning every ambient direction for each column.
    same_algebra = n > 1 && all(j -> left[j].algebra === plan.algebra &&
        right[j].algebra === plan.algebra, eachindex(left, right))
    same_algebra && _validate_plan_algebra(plan, plan.algebra)
    for j in 1:n
        same_algebra || _validate_plan_algebra(plan, _same_algebra(left[j], right[j]))
        if haskey(seen_left, left[j])
            copyto!(view(left_values, :, j),
                    view(left_values, :, seen_left[left[j]]))
        else
            _matches_plan_support(left[j], plan.left_support) ||
                throw(ArgumentError("left support differs from prepared product"))
            for (i, mask) in enumerate(plan.left_masks)
                left_values[i, j] = coefficient_mask(left[j], mask)
            end
            seen_left[left[j]] = j
        end
        if haskey(seen_right, right[j])
            copyto!(view(right_values, :, j),
                    view(right_values, :, seen_right[right[j]]))
        else
            _matches_plan_support(right[j], plan.right_support) ||
                throw(ArgumentError("right support differs from prepared product"))
            for (i, mask) in enumerate(plan.right_masks)
                right_values[i, j] = coefficient_mask(right[j], mask)
            end
            seen_right[right[j]] = j
        end
    end
    same_algebra && _validate_plan_algebra(plan, plan.algebra)
    return PackedProductBatch(plan, left_values, right_values)
end

"""Return output coefficient rows by plan mask, columns by batch item."""
function run_packed_batch(batch::PackedProductBatch{S}) where S
    plan = batch.plan
    isdiag(metric(plan.algebra)) &&
        diag(metric(plan.algebra)) == plan.diagonal &&
        basis(plan.algebra) == plan.basis_names ||
        throw(ArgumentError("packed product plan's algebra changed"))
    n = size(batch.left_values, 2)
    output = zeros(S, length(plan.output_masks), n)
    for j in 1:n
        for (ai, bi, oi, factor) in plan.paths
            @inbounds output[oi, j] += factor * batch.left_values[ai, j] *
                                      batch.right_values[bi, j]
        end
    end
    return output
end

"""Convert packed output columns back to sparse multivectors when needed."""
function unpack_product_batch(batch::PackedProductBatch,
                              values::AbstractMatrix)
    plan = batch.plan
    size(values) == (length(plan.output_masks), size(batch.left_values, 2)) ||
        throw(DimensionMismatch("packed output shape does not match plan and batch"))
    ga = plan.algebra
    K = eltype(plan.output_masks)
    T = eltype(values)
    return [SparseMultiVector(ga,
                Dict{K,T}(mask => value for (mask, value) in
                          zip(plan.output_masks, view(values, :, j))
                          if !iszero(value)))
            for j in axes(values, 2)]
end
