# Exact recursively synthesized bilinear matrix kernels for Cl(r,r).

struct BilinearSplitPlan{A<:GeometricAlgebra}
    algebra::A
    left::Matrix{BigInt}
    cutoff::Int
    max_bytes::Int
end

function _strassen_product(a::AbstractMatrix{BigInt},b::AbstractMatrix{BigInt},
                           cutoff::Int)
    side=size(a,1)
    side<=cutoff && return a*b
    half=side÷2
    a11=@view a[1:half,1:half]; a12=@view a[1:half,half+1:side]
    a21=@view a[half+1:side,1:half]; a22=@view a[half+1:side,half+1:side]
    b11=@view b[1:half,1:half]; b12=@view b[1:half,half+1:side]
    b21=@view b[half+1:side,1:half]; b22=@view b[half+1:side,half+1:side]
    m1=_strassen_product(a11+a22,b11+b22,cutoff)
    m2=_strassen_product(a21+a22,b11,cutoff)
    m3=_strassen_product(a11,b12-b22,cutoff)
    m4=_strassen_product(a22,b21-b11,cutoff)
    m5=_strassen_product(a11+a12,b22,cutoff)
    m6=_strassen_product(a21-a11,b11+b12,cutoff)
    m7=_strassen_product(a12-a22,b21+b22,cutoff)
    [m1+m4-m5+m7 m3+m5; m2+m4 m1-m2+m3+m6]
end

"""Prepare a recursively composed seven-product bilinear split kernel.

The exact BigInt contract is bounded to Cl(r,r), 1<=r<=5. A cutoff
chooses where classical matrix multiplication replaces recursion.
"""
function prepare_bilinear_split(left::DenseMultiVector{BigInt};
                                cutoff::Int=1,max_bytes::Int=512<<20)
    side=1<<(dimension(left.algebra)÷2)
    1<=cutoff<=side && ispow2(cutoff) ||
        throw(ArgumentError("bilinear cutoff must be a power of two within matrix side"))
    matrix=split_matrix_transform(left;max_bytes)
    BilinearSplitPlan(left.algebra,matrix,cutoff,max_bytes)
end

"""Run the exact bilinear kernel and invert the matrix representation."""
function run_bilinear_split(plan::BilinearSplitPlan,
                            right::DenseMultiVector{BigInt})
    _split_same_algebra(plan.algebra,right.algebra) ||
        throw(ArgumentError("bilinear operand algebra mismatch"))
    rhs=split_matrix_transform(right;max_bytes=plan.max_bytes)
    product=_strassen_product(plan.left,rhs,plan.cutoff)
    split_matrix_inverse(plan.algebra,product;max_bytes=plan.max_bytes)
end

function bilinear_split_stats(plan::BilinearSplitPlan)
    side=size(plan.left,1)
    depth=trailing_zeros(side)-trailing_zeros(plan.cutoff)
    (side=side,cutoff=plan.cutoff,recursive_levels=depth,
     scalar_products=7^depth*plan.cutoff^3,
     retained_matrix_cells=length(plan.left))
end

# The 2x2 tensor certificate tests every elementary left/right matrix pair.
# Bilinearity then proves equality on all matrices over any commutative ring.
function _verify_split_kernel(kernel::Symbol)
    kernel in (:classical,:strassen) ||
        throw(ArgumentError("unknown bilinear candidate"))
    for left_index in 1:4,right_index in 1:4
        a=zeros(BigInt,2,2);b=zeros(BigInt,2,2)
        a[left_index]=1;b[right_index]=1
        obtained=kernel==:classical ? a*b : _strassen_product(a,b,1)
        obtained==a*b || return false
    end
    true
end

function _split_kernel_cost(side::Int,kernel::Symbol,mul_weight::Int,
                            add_weight::Int)
    if kernel==:classical || side==1
        return mul_weight*side^3+add_weight*side^2*(side-1)
    end
    half=side÷2
    7*_split_kernel_cost(half,:strassen,mul_weight,add_weight)+
        18*add_weight*half^2
end

"""A proof-carrying bounded search over classical and Strassen kernels.

The base 2x2 bilinear tensors are checked on all elementary pairs. Recursive
composition preserves the certificate. The cost weights affect selection,
never arithmetic correctness.
"""
struct VerifiedBilinearPlan{P<:BilinearSplitPlan}
    candidate::Symbol
    plan::P
    verified_pairs::Int
    estimated_cost::Int
    mul_weight::Int
    add_weight::Int
end

function prepare_verified_bilinear(left::DenseMultiVector{BigInt};
        candidates::Tuple{Vararg{Symbol}}=(:classical,:strassen),
        mul_weight::Int=10,add_weight::Int=1,max_bytes::Int=512<<20)
    isempty(candidates) && throw(ArgumentError("at least one kernel candidate is required"))
    mul_weight>=1 && add_weight>=1 ||
        throw(ArgumentError("kernel cost weights must be positive"))
    all(_verify_split_kernel,candidates) ||
        throw(ArgumentError("bilinear candidate failed tensor verification"))
    side=1<<(dimension(left.algebra)÷2)
    ranked=[(_split_kernel_cost(side,k,mul_weight,add_weight),k)
        for k in candidates]
    sort!(ranked;by=first)
    cost,candidate=first(ranked)
    cutoff=candidate==:classical ? side : 1
    plan=prepare_bilinear_split(left;cutoff,max_bytes)
    VerifiedBilinearPlan(candidate,plan,16*length(unique(candidates)),
                         cost,mul_weight,add_weight)
end

run_verified_bilinear(plan::VerifiedBilinearPlan,
                      right::DenseMultiVector{BigInt})=
    run_bilinear_split(plan.plan,right)

verified_bilinear_stats(plan::VerifiedBilinearPlan)=(
    candidate=plan.candidate,verified_pairs=plan.verified_pairs,
    estimated_cost=plan.estimated_cost,mul_weight=plan.mul_weight,
    add_weight=plan.add_weight)
