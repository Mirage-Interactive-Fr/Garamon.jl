"""
    outermorphism(P, a, target; max_terms=65536, check_metric=false)

Apply the exterior extension of the linear map whose columns are the images
of the source basis vectors in `target`. The map may be rectangular or singular.
Set `check_metric=true` to require `transpose(P) * metric(target) * P ≈ metric(source)`;
only then is it also a Clifford-algebra homomorphism.
"""
function outermorphism(P::AbstractMatrix, a::AbstractMultiVector,
                       target::GeometricAlgebra;
                       max_terms::Integer=1 << 16,
                       check_metric::Bool=false)
    source = a.algebra
    size(P) == (dimension(target), dimension(source)) ||
        throw(DimensionMismatch("linear map must have target rows and source columns"))
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    if check_metric && !isapprox(transpose(P) * metric(target) * P, metric(source))
        throw(ArgumentError("linear map does not preserve the metric"))
    end
    T = promote_type(eltype(P), eltype(metric(target)), eltype(a))
    K = _masktype(target)
    result = Dict{K,T}()
    for (mask, value) in _terms(a)
        current = Dict{K,T}(zero(K) => convert(T, value))
        for col in _indices(source, mask)
            next = Dict{K,T}()
            for (blade, coefficient) in current, row in 1:dimension(target)
                x = P[row, col]
                iszero(x) && continue
                bit = one(K) << (row - 1)
                iszero(blade & bit) || continue
                lower = bit - one(K)
                sign = isodd(_blade_grade(blade & ~lower)) ? -1 : 1
                _addterm!(next, blade | bit, sign * coefficient * x)
                length(next) <= max_terms ||
                    throw(ArgumentError("outermorphism expansion exceeds max_terms"))
            end
            length(next) <= max_terms ||
                throw(ArgumentError("outermorphism expansion exceeds max_terms"))
            current = next
        end
        for (blade, coefficient) in current
            _addterm!(result, blade, coefficient)
            length(result) <= max_terms ||
                throw(ArgumentError("outermorphism expansion exceeds max_terms"))
        end
        length(result) <= max_terms ||
            throw(ArgumentError("outermorphism expansion exceeds max_terms"))
    end
    return SparseMultiVector(target, result)
end

"""
    FactorizedBlade(algebra, factors; scale=1)

A simple blade `scale * factors[:,1] ∧ ⋯ ∧ factors[:,k]`. This representation
keeps `n*k` entries instead of `binomial(n,k)` blade coefficients.
"""
struct FactorizedBlade{T,A<:GeometricAlgebra}
    algebra::A
    factors::Matrix{T}
    scale::T
    function FactorizedBlade(ga::A, factors::AbstractMatrix{T};
                             scale=one(T)) where {A<:GeometricAlgebra,T}
        size(factors, 1) == dimension(ga) ||
            throw(DimensionMismatch("factors must have one row per basis vector"))
        S = promote_type(T, typeof(scale))
        new{S,A}(ga, Matrix{S}(factors), convert(S, scale))
    end
end

function coefficient(blade::FactorizedBlade{T}, indices) where T
    mask = _mask(blade.algebra, indices)
    rows = _indices(blade.algebra, mask)
    length(rows) == size(blade.factors, 2) || return zero(T)
    isempty(rows) && return blade.scale
    exact = T <: Integer || T <: Rational || T <: Complex{<:Integer} ||
            T <: Complex{<:Rational}
    S = exact ? (T <: Real ? Rational{BigInt} : Complex{Rational{BigInt}}) : T
    return blade.scale * det(Matrix{S}(blade.factors[rows, :]))
end

function wedge(a::FactorizedBlade, b::FactorizedBlade)
    dimension(a.algebra) == dimension(b.algebra) &&
        metric(a.algebra) == metric(b.algebra) &&
        basis(a.algebra) == basis(b.algebra) ||
        throw(ArgumentError("blades belong to different algebras or bases"))
    return FactorizedBlade(a.algebra, hcat(a.factors, b.factors);
                           scale=a.scale * b.scale)
end
∧(a::FactorizedBlade, b::FactorizedBlade) = wedge(a, b)

function outermorphism(P::AbstractMatrix, blade::FactorizedBlade,
                       target::GeometricAlgebra; check_metric::Bool=false)
    size(P) == (dimension(target), dimension(blade.algebra)) ||
        throw(DimensionMismatch("linear map must have target rows and source columns"))
    if check_metric && !isapprox(transpose(P) * metric(target) * P, metric(blade.algebra))
        throw(ArgumentError("linear map does not preserve the metric"))
    end
    return FactorizedBlade(target, P * blade.factors; scale=blade.scale)
end

"""Expand a factorized blade only while its active support stays within `max_terms`."""
function expand(blade::FactorizedBlade; max_terms::Integer=1 << 16)
    max_terms >= 1 || throw(ArgumentError("max_terms must be positive"))
    ga = blade.algebra
    T = promote_type(eltype(blade.factors), eltype(metric(ga)))
    K = _masktype(ga)
    current = Dict{K,T}(zero(K) => convert(T, blade.scale))
    for col in axes(blade.factors, 2)
        next = Dict{K,T}()
        for (mask, value) in current, row in 1:dimension(ga)
            x = blade.factors[row, col]
            iszero(x) && continue
            bit = one(K) << (row - 1)
            iszero(mask & bit) || continue
            sign = isodd(_blade_grade(mask & ~(bit - one(K)))) ? -1 : 1
            _addterm!(next, mask | bit, sign * value * x)
            length(next) <= max_terms ||
                throw(ArgumentError("factorized blade expansion exceeds max_terms"))
        end
        length(next) <= max_terms ||
            throw(ArgumentError("factorized blade expansion exceeds max_terms"))
        current = next
    end
    return SparseMultiVector(ga, current)
end

"""
    ReflectionChain(algebra, normals)

Store non-null reflection normals as columns, in application order. The action
is `v ↦ v - 2*g(v,u)/g(u,u)*u`, equivalent to `-u*v*inv(u)` for a vector `u`.
Several reflections implement the associated factorized versor action without
forming a multivector with up to `2^n` coefficients.
"""
struct ReflectionChain{T,A<:GeometricAlgebra}
    algebra::A
    normals::Matrix{T}
    norms::Vector{T}
    function ReflectionChain(ga::A, normals::AbstractMatrix{U}) where
            {A<:GeometricAlgebra,U}
        size(normals, 1) == dimension(ga) ||
            throw(DimensionMismatch("normals must have one row per basis vector"))
        S0 = promote_type(U, eltype(metric(ga)))
        S = S0 <: Integer ? Rational{BigInt} : S0
        vectors = Matrix{S}(normals)
        norms = S[]
        for col in axes(vectors, 2)
            normal = view(vectors, :, col)
            squared = sum(normal .* (metric(ga) * normal))
            iszero(squared) && throw(DomainError(normal, "reflection normal is null"))
            push!(norms, squared)
        end
        new{S,A}(ga, vectors, norms)
    end
end

"""Apply a reflection chain to the coordinates of a grade-one vector."""
function versor_action(chain::ReflectionChain{T}, vector::AbstractVector) where T
    length(vector) == dimension(chain.algebra) ||
        throw(DimensionMismatch("vector dimension differs from algebra"))
    S = promote_type(T, eltype(vector))
    result = Vector{S}(vector)
    gram = metric(chain.algebra)
    small_diagonal = length(result) <= 16 && isdiag(gram)
    for col in axes(chain.normals, 2)
        pairing = if small_diagonal
            value = zero(S)
            for row in eachindex(result)
                value += chain.normals[row, col] * (gram[row, row] * result[row])
            end
            value
        else
            normal = view(chain.normals, :, col)
            sum(normal .* (gram * result))
        end
        factor = 2 * pairing / chain.norms[col]
        for row in eachindex(result)
            result[row] -= factor * chain.normals[row, col]
        end
    end
    return result
end

function versor_action(chain::ReflectionChain, blade::FactorizedBlade)
    dimension(chain.algebra) == dimension(blade.algebra) &&
        metric(chain.algebra) == metric(blade.algebra) &&
        basis(chain.algebra) == basis(blade.algebra) ||
        throw(ArgumentError("blade belongs to a different algebra or basis"))
    transformed = isempty(axes(blade.factors, 2)) ?
        zeros(promote_type(eltype(chain.normals), eltype(blade.factors)),
              dimension(chain.algebra), 0) :
        hcat((versor_action(chain, view(blade.factors, :, j))
              for j in axes(blade.factors, 2))...)
    return FactorizedBlade(chain.algebra, transformed; scale=blade.scale)
end
