struct GeometricAlgebra{T<:Number,M<:AbstractMatrix{T}}
    metric::M
    basis::Vector{String}
    kind::Symbol
    multiplicity::Int
    object_dim::Int
end

metric(ga) = ga.metric
basis(ga) = ga.basis
kind(ga) = ga.kind
multiplicity(ga) = ga.multiplicity
object_dim(ga) = ga.object_dim

dimension(metric) = size(metric, 1)
dimension(ga::GeometricAlgebra) = dimension(metric(ga))

single_metric(m::AbstractMatrix, _) = m
single_metric(space_dim, kind) = single_metric(space_dim, Val(kind))

function multiple_metric(metric, multiplicity)
    multiplicity == 1 && return metric
    n = dimension(metric)
    return kron(spdiagm(0 => ones(eltype(metric), multiplicity)), sparse(metric))
end

max_object_dim(metric, object_dim) = metric

function algebra(
    metric_or_space_dim,
    kind = :none;
    multiplicity = 1,
    object_dim = 1,
    basis = Vector{String}()
)
    multiplicity >= 1 || throw(ArgumentError("multiplicity must be positive"))
    object_dim >= 1 || throw(ArgumentError("object_dim must be positive"))
    inner_kind = kind == :none ? :ega : kind
    metric = single_metric(metric_or_space_dim, inner_kind)
    size(metric, 1) == size(metric, 2) || throw(DimensionMismatch("metric must be square"))
    dimension(metric) >= 1 || throw(ArgumentError("metric must have positive dimension"))
    issymmetric(metric) || throw(ArgumentError("metric must be symmetric"))
    metric = multiple_metric(metric, multiplicity)
    metric = max_object_dim(metric, object_dim)
    inner_basis = if isempty(basis)
        basis_vectors_names(metric, multiplicity, object_dim, Val(inner_kind))
    else
        length(basis) == dimension(metric) || throw(DimensionMismatch("basis names must match metric dimension"))
        length(unique(basis)) == length(basis) || throw(ArgumentError("basis names must be distinct"))
        basis
    end
    return GeometricAlgebra(metric, inner_basis, kind, multiplicity, object_dim)
end

limit_bases_indices(kind) = limit_bases_indices(Val(kind))

function delimiter_indices(metric, multiplicity, object_dim, kind)
    λ = 9 + multiplicity * limit_bases_indices(kind)
    return dimension(metric) < λ ? "" : "_"
end

function basis_vectors_names(ga)
    m, μ, d = metric(ga), multiplicity(ga), object_dim(ga)
    return basis_vectors_names(m, μ, d, Val(kind(ga)))
end
space_dimension(dim, ::Val) = dim
space_dimension(ga) = space_dimension(div(dimension(ga), multiplicity(ga)), Val(kind(ga)))

function descriptor(ga)
    pre = kind(ga) == :none ? "my" : string(kind(ga))[1:end-2]
    spd = space_dimension(ga)
    str = "ga"
    mul = multiplicity(ga) == 1 ? "" : string(multiplicity(ga))
    obj = object_dim(ga) == 1 ? "" : "_$(object_dim(ga))"
    return "$pre$spd$str$mul$obj"
end

# SECTION - Euclidean Geometric Algebra
limit_bases_indices(::Val{:ega}) = 0

function basis_vectors_names(m, μ, d, ::Val{:ega})
    di = delimiter_indices(m, μ, d, :ega)
    return ["$di$d" for d in 1:dimension(m)]
end

function single_metric(space_dim::Integer, ::Val{:ega})
    space_dim >= 1 || throw(ArgumentError("space dimension must be positive"))
    return space_dim <= 8 ?
        SMatrix{space_dim,space_dim}(Matrix{Float64}(I, space_dim, space_dim)) :
        spdiagm(0 => ones(Float64, space_dim))
end

# SECTION - Conformal Geometric Algebra

limit_bases_indices(::Val{:cga}) = 2

function basis_vectors_names(m, μ, d, ::Val{:cga})
    di = delimiter_indices(m, μ, d, :cga)
    names = Vector{String}()
    for i in 1:μ
        for j in 0:(dimension(m) ÷ μ - 2)
            d = j == 0 && μ > 1 ? string(i) : ""
            push!(names, "$di$j$d")
        end
        d = μ > 1 ? string(i) : ""
        push!(names, "i$i")
    end
    return names
end

function single_metric(space_dim, ::Val{:cga})
    dim = space_dim + 2
    if dim > 8
        result = spdiagm(0 => [1 < i < dim ? 1.0 : 0.0 for i in 1:dim])
        result[1, dim] = result[dim, 1] = -1.0
        return result
    end
    f = (i, j) -> 1 < i == j < dim ? 1.0 :
                  ((i, j) == (1, dim) || (i, j) == (dim, 1)) ? -1.0 : 0.0
    return SMatrix{dim,dim}([f(i, j) for i in 1:dim, j in 1:dim])
end

space_dimension(dim, ::Val{:cga}) = dim - 2

# SECTION - Projective Geometric Algebra

limit_bases_indices(::Val{:pga}) = 1

function basis_vectors_names(m, μ, d, ::Val{:pga})
    di = delimiter_indices(m, μ, d, :pga)
    return ["$di$d" for d in 0:(dimension(m)-1)]
end

function single_metric(space_dim, ::Val{:pga})
    dim = space_dim + 1
    if dim > 8
        return spdiagm(0 => [i == 1 ? 0.0 : 1.0 for i in 1:dim])
    end
    f = (i, j) -> 1 < i == j ≤ dim ? 1.0 : 0.0
    return SMatrix{dim,dim}([f(i, j) for i in 1:dim, j in 1:dim])
end

# SECTION - Projective Space Geometric Algebra

limit_bases_indices(::Val{:psga}) = 9

function basis_vectors_names(m, μ, d, ::Val{:psga})
    di = delimiter_indices(m, μ, d, :psga)
    inds = Vector{String}()
    dimdiv2 = round(Int, dimension(m) / 2)
    for d in ["", "d"], i in 1:dimdiv2
        push!(inds, "$d$di$i")
    end
    return inds
end

function single_metric(space_dim::Int, ::Val{:psga})
    dim = space_dim + 1
    if 2dim > 8
        result = spzeros(Float64, 2dim, 2dim)
        for i in 1:dim
            result[i, i + dim] = result[i + dim, i] = 0.5
        end
        return result
    end
    f = (i, j) -> abs(i - j) == dim ? 0.5 : 0.0
    return SMatrix{2dim,2dim}([f(i, j) for i in 1:2dim, j in 1:2dim])
end

# SECTION - Table of different GA
const GEOMETRIC_ALGEBRAS = Dict(
    :cga => "Conformal Geometric Algebra",
    :ega => "Euclidean Geometric Algebra",
    :pga => "Projective Geometric Algebra",
    :psga => "Projective Space Geometric Algebra",
)

list_geometric_algebras(list=GEOMETRIC_ALGEBRAS) = pretty_table(list)
