"""One exact elementary number-conserving fermionic Gaussian shear."""
struct FermionicShear
    target::Int
    source::Int
    coefficient::BigInt
end

"""A sequence of quadratic-generator shears on an n-mode Fock space."""
struct FermionicGaussianPlan{M}
    dimension::Int
    shears::Vector{FermionicShear}
    matrix::M
end

function _fermionic_shears(n::Int,specifications)
    shears=FermionicShear[]
    for specification in specifications
        length(specification)==3 ||
            throw(ArgumentError("a shear needs (target,source,coefficient)"))
        target,source,coefficient=specification
        target isa Integer && source isa Integer && coefficient isa Integer ||
            throw(ArgumentError("fermionic shear entries must be integers"))
        1<=target<=n && 1<=source<=n && target!=source ||
            throw(ArgumentError("invalid fermionic shear coordinates"))
        push!(shears,FermionicShear(Int(target),Int(source),BigInt(coefficient)))
    end
    shears
end

function _fermionic_transform!(output::Matrix{BigInt},shears)
    for shear in shears
        for j in axes(output,2)
            output[shear.target,j]+=shear.coefficient*output[shear.source,j]
        end
    end
    output
end

"""
    prepare_fermionic_gaussian(n,shears;materialize=false)

Each (i,j,c) is the exact second-quantized one-body map
exp(c a_i^dagger a_j): row i += c*row j, i != j. This is a
number-conserving, generally nonunitary Gaussian subclass. The Fock-space
input is a decomposable Slater state, supplied as an n-by-k orbital matrix.
"""
function prepare_fermionic_gaussian(n::Integer,specifications;
        materialize::Bool=false,max_shears::Integer=1<<16)
    1<=n<=typemax(Int) || throw(ArgumentError("invalid mode dimension"))
    0<=max_shears<=typemax(Int) || throw(ArgumentError("invalid shear budget"))
    length(specifications)<=max_shears ||
        throw(ArgumentError("fermionic shear budget exceeded"))
    shears=_fermionic_shears(Int(n),specifications)
    matrix=if materialize
        identity=zeros(BigInt,Int(n),Int(n))
        for i in 1:Int(n)
            identity[i,i]=1
        end
        _fermionic_transform!(identity,shears)
    else
        nothing
    end
    FermionicGaussianPlan(Int(n),shears,matrix)
end

function _fermionic_orbitals(plan::FermionicGaussianPlan,
        occupied::AbstractMatrix)
    size(occupied,1)==plan.dimension ||
        throw(DimensionMismatch("occupied orbitals must match mode dimension"))
    eltype(occupied)<:Integer ||
        throw(ArgumentError("fermionic orbitals must have integer coordinates"))
    input=Matrix{BigInt}(occupied)
    plan.matrix===nothing ? _fermionic_transform!(input,plan.shears) :
        plan.matrix*input
end

"""Return exact overlap with a target Slater state after the Gaussian map."""
function run_fermionic_gaussian(plan::FermionicGaussianPlan,
        occupied::AbstractMatrix,target::AbstractMatrix;
        diagnostics::Bool=false)
    size(target,1)==plan.dimension &&
        size(target,2)==size(occupied,2) ||
        throw(DimensionMismatch("target orbitals must have the same mode dimension and grade"))
    eltype(target)<:Integer ||
        throw(ArgumentError("target orbitals must have integer coordinates"))
    transformed=_fermionic_orbitals(plan,occupied)
    target_exact=Matrix{BigInt}(target)
    gram=transpose(target_exact)*transformed
    rank,overlap=_cross_gram_rank_det(gram)
    diagnostics ? (overlap,(rank=rank,
        representation=plan.matrix===nothing ? :shears : :matrix,
        matrix_entries=plan.matrix===nothing ? 0 : length(plan.matrix))) : overlap
end
