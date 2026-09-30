"""Structural certificates for both steps of an exact three-input product."""
struct PropagatedProductCertificate{P1<:ProductPlan,P2<:ProductPlan}
    first::P1
    second::P2
    max_paths::Int
end

"""
    prepare_propagated_product(a,b,c; max_paths=65536)

Prepare (a*b)*c for repeated integer inputs on a diagonal metric whose
entries are -1, 0, or 1. The first plan's possible output masks become the
second plan's certified input support. Actual cancellation may shrink that
support without invalidating the second plan.
"""
function prepare_propagated_product(a::AbstractMultiVector,
        b::AbstractMultiVector,c::AbstractMultiVector;
        max_paths::Integer=1<<16)
    ga=_same_algebra(a,b)
    _same_algebra(a,c)
    eltype(a)<:Integer && eltype(b)<:Integer && eltype(c)<:Integer &&
        eltype(metric(ga))<:Integer && isdiag(metric(ga)) &&
        all(metric(ga)[i,i] in (-1,0,1) for i in 1:dimension(ga)) ||
        throw(ArgumentError("propagated certificate requires integer inputs and a diagonal metric in {-1,0,1}"))
    0<=max_paths<=typemax(Int) ||
        throw(ArgumentError("invalid propagated path budget"))
    first=prepare_product(a,b;max_paths)
    K=_masktype(ga)
    potential=multivector(ga,Dict{K,BigInt}(
        mask=>BigInt(1) for mask in first.output_masks);storage=:sparse)
    second=prepare_product(potential,c;max_paths)
    PropagatedProductCertificate(first,second,Int(max_paths))
end

function _propagated_valid(certificate::PropagatedProductCertificate,
        a::AbstractMultiVector,b::AbstractMultiVector,c::AbstractMultiVector)
    ga=_same_algebra(a,b)
    _same_algebra(a,c)
    _validate_plan_inputs(certificate.first,a,b)
    _validate_plan_algebra(certificate.second,ga)
    _matches_plan_support(c,certificate.second.right_support) ||
        throw(ArgumentError("third input support differs from the certificate"))
    true
end

"""
    run_propagated_product(cert,a,b,c; on_invalid=:error,diagnostics=false)

Execute both certified plans. The second allows a subset of its predicted
intermediate support. With on_invalid=:direct, failed validation triggers
a complete exact direct product, never a truncated certified result.
"""
function run_propagated_product(cert::PropagatedProductCertificate,
        a::AbstractMultiVector,b::AbstractMultiVector,c::AbstractMultiVector;
        on_invalid::Symbol=:error,diagnostics::Bool=false)
    on_invalid in (:error,:direct) ||
        throw(ArgumentError("on_invalid must be :error or :direct"))
    valid=try
        _propagated_valid(cert,a,b,c)
    catch exception
        exception isa ArgumentError || rethrow()
        false
    end
    if !valid
        on_invalid==:error &&
            throw(ArgumentError("propagated product certificate is invalid"))
        result=geometric_product(geometric_product(a,b),c)
        return diagnostics ? (result,(used_fallback=true,
            predicted_support=length(cert.second.left_masks))) : result
    end
    intermediate=run_product(cert.first,a,b)
    result=run_product(cert.second,intermediate,c;allow_subsets=true)
    diagnostics ? (result,(used_fallback=false,
        predicted_support=length(cert.second.left_masks),
        actual_support=length(intermediate.values))) : result
end
