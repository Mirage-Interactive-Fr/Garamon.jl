"""A coordinate subalgebra selected by the active basis directions."""
struct CoordinateSubspace{T,A<:GeometricAlgebra}
    ambient_dimension::Int
    indices::Vector{Int}
    restricted_metric::Matrix{T}
    selected_names::Vector{String}
    algebra::A
end

"""
    coordinate_subspace(a, b)

Construct the smallest coordinate subalgebra containing both supports. Its
metric is the principal submatrix on the active directions. Scalar-only
operands use one arbitrary ambient direction because algebras have positive
dimension. Setup is explicit and can be reused for later operands whose
supports remain inside the selected directions.
"""
function coordinate_subspace(a::AbstractMultiVector, b::AbstractMultiVector)
    ga = _same_algebra(a, b)
    selected = Set{Int}()
    for mv in (a, b), (mask, _) in _terms(mv)
        union!(selected, _indices(ga, mask))
    end
    indices = isempty(selected) ? [1] : sort!(collect(selected))
    restricted = Matrix(metric(ga)[indices, indices])
    names = copy(basis(ga)[indices])
    local_algebra = algebra(restricted; basis=names)
    return CoordinateSubspace(dimension(ga), indices, restricted,
                              names, local_algebra)
end

function _check_subspace(plan::CoordinateSubspace, ga::GeometricAlgebra)
    dimension(ga) == plan.ambient_dimension &&
        metric(ga)[plan.indices, plan.indices] == plan.restricted_metric &&
        basis(ga)[plan.indices] == plan.selected_names ||
        throw(ArgumentError("coordinate subspace no longer matches the algebra"))
end

"""Map a multivector with support inside a coordinate subspace to local masks."""
function project_subspace(plan::CoordinateSubspace, a::AbstractMultiVector)
    _check_subspace(plan, a.algebra)
    lookup = Dict(i => j for (j, i) in enumerate(plan.indices))
    K = _masktype(plan.algebra)
    T = promote_type(eltype(a), eltype(metric(plan.algebra)))
    result = Dict{K,T}()
    for (mask, value) in _terms(a)
        local_mask = zero(K)
        for i in _indices(a.algebra, mask)
            j = get(lookup, i, 0)
            iszero(j) && throw(ArgumentError("input support exceeds coordinate subspace"))
            local_mask |= one(K) << (j - 1)
        end
        result[local_mask] = convert(T, value)
    end
    return SparseMultiVector(plan.algebra, result)
end

"""Embed a local result back into a compatible ambient algebra."""
function lift_subspace(plan::CoordinateSubspace, a::AbstractMultiVector,
                       target::GeometricAlgebra)
    _check_subspace(plan, target)
    dimension(a.algebra) == length(plan.indices) &&
        metric(a.algebra) == plan.restricted_metric &&
        basis(a.algebra) == plan.selected_names ||
        throw(ArgumentError("result belongs to a different coordinate subspace"))
    K = _masktype(target)
    T = promote_type(eltype(a), eltype(metric(target)))
    result = Dict{K,T}()
    for (mask, value) in _terms(a)
        ambient_mask = zero(K)
        for j in _indices(a.algebra, mask)
            ambient_mask |= one(K) << (plan.indices[j] - 1)
        end
        result[ambient_mask] = convert(T, value)
    end
    return SparseMultiVector(target, result)
end

"""Execute a product in a preselected coordinate subalgebra and lift it."""
function subspace_product(plan::CoordinateSubspace, a::AbstractMultiVector,
                          b::AbstractMultiVector;
                          operation::Symbol=:geometric,
                          max_terms::Integer=1 << 16)
    ga = _same_algebra(a, b)
    local_a = project_subspace(plan, a)
    local_b = project_subspace(plan, b)
    return lift_subspace(plan, _product(local_a, local_b, operation;
                                        max_terms), ga)
end
