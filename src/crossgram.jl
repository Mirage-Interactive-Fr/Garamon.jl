"""An exact cross-Gram certificate for one scalar pairing of decomposable blades."""
struct CrossGramPlan
    diagonal::Vector{BigInt}
    left::Matrix{BigInt}
    right::Matrix{BigInt}
    gram::Matrix{BigInt}
    rank::Int
    determinant::BigInt
end

function _cross_gram_inputs(ga::GeometricAlgebra,left::AbstractMatrix,
        right::AbstractMatrix)
    n=dimension(ga)
    size(left,1)==n && size(right,1)==n ||
        throw(DimensionMismatch("vector coordinates must match algebra dimension"))
    size(left,2)==size(right,2) ||
        throw(DimensionMismatch("blade grades must agree"))
    size(left,2)<=16 || throw(ArgumentError("cross-Gram grade exceeds 16"))
    eltype(left)<:Integer && eltype(right)<:Integer &&
        eltype(metric(ga))<:Integer && isdiag(metric(ga)) ||
        throw(ArgumentError("cross-Gram requires integer vectors and diagonal integer metric"))
    diagonal=BigInt[metric(ga)[i,i] for i in 1:n]
    diagonal,Matrix{BigInt}(left),Matrix{BigInt}(right)
end

function _cross_gram_matrix(diagonal,left,right)
    n,k=size(left)
    gram=zeros(BigInt,k,k)
    for j in 1:k,i in 1:k,l in 1:n
        gram[i,j]+=left[l,i]*diagonal[l]*right[l,j]
    end
    gram
end

function _cross_gram_rank_det(gram::Matrix{BigInt})
    k=size(gram,1)
    work=Rational{BigInt}.(gram)
    rank=0
    sign=1
    determinant=one(Rational{BigInt})
    for col in 1:k
        pivot=findfirst(row->!iszero(work[row,col]),rank+1:k)
        pivot===nothing && continue
        row=rank+pivot
        rank+=1
        if row!=rank
            work[rank,:],work[row,:]=copy(work[row,:]),copy(work[rank,:])
            sign=-sign
        end
        value=work[rank,col]
        determinant*=value
        for target in rank+1:k
            factor=work[target,col]/value
            iszero(factor) && continue
            for j in col:k
                work[target,j]-=factor*work[rank,j]
            end
        end
    end
    rank==k || return rank,BigInt(0)
    result=sign*determinant
    denominator(result)==1 || error("integer Gram matrix has noninteger determinant")
    rank,numerator(result)
end

"""
    prepare_cross_gram(ga,left,right)

For two equal-grade decomposable blades, return a reusable certificate of
their reverse-product scalar pairing. A deficient exact cross-Gram rank
certifies a zero pairing; no coefficient of a general product is inferred.
"""
function prepare_cross_gram(ga::GeometricAlgebra,left::AbstractMatrix,
        right::AbstractMatrix)
    diagonal,u,v=_cross_gram_inputs(ga,left,right)
    gram=_cross_gram_matrix(diagonal,u,v)
    rank,determinant=_cross_gram_rank_det(gram)
    CrossGramPlan(diagonal,u,v,gram,rank,determinant)
end

"""Return the exact scalar pairing; changed inputs can trigger recomputation."""
function run_cross_gram(plan::CrossGramPlan,ga::GeometricAlgebra,
        left::AbstractMatrix,right::AbstractMatrix;
        left_scale::Integer=1,right_scale::Integer=1,
        on_invalid::Symbol=:error,diagnostics::Bool=false)
    on_invalid in (:error,:direct) ||
        throw(ArgumentError("on_invalid must be :error or :direct"))
    diagonal,u,v=_cross_gram_inputs(ga,left,right)
    valid=diagonal==plan.diagonal && u==plan.left && v==plan.right
    if !valid
        on_invalid==:error && throw(ArgumentError("cross-Gram certificate is invalid"))
        gram=_cross_gram_matrix(diagonal,u,v)
        rank,determinant=_cross_gram_rank_det(gram)
        result=BigInt(left_scale)*BigInt(right_scale)*determinant
        return diagnostics ? (result,(used_fallback=true,rank=rank)) : result
    end
    result=BigInt(left_scale)*BigInt(right_scale)*plan.determinant
    diagnostics ? (result,(used_fallback=false,rank=plan.rank)) : result
end

"""Compute an exact decomposable-blade scalar pairing without a saved plan."""
cross_gram_scalar(ga::GeometricAlgebra,left::AbstractMatrix,
    right::AbstractMatrix;left_scale::Integer=1,right_scale::Integer=1)=
    run_cross_gram(prepare_cross_gram(ga,left,right),ga,left,right;
        left_scale,right_scale)
