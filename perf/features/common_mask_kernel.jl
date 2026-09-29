using Garamon

function mask_kernel_state(n::Int, representation::Symbol, workload::Symbol)
    64 <= n <= 128 || throw(ArgumentError("mask experiment covers 64:128"))
    K = representation == :fixed ? UInt128 :
        representation == :arbitrary ? BigInt :
        throw(ArgumentError("unknown representation"))
    blade(i, j) = (one(K) << (i - 1)) | (one(K) << (j - 1))
    masks = K[zero(K), blade(1, 2), blade(1, n), blade(n - 1, n)]
    for i in 2:min(n - 2, 10)
        push!(masks, blade(i, n - i))
    end
    unique!(masks)
    left = Dict{K,Float64}(mask => Float64(mod(i, 5) + 1)
                           for (i, mask) in enumerate(masks))
    right = Dict{K,Float64}(mask => Float64(mod(2i, 7) + 1)
                            for (i, mask) in enumerate(reverse(masks)))
    target = workload == :targeted ? blade(1, n) : zero(K)
    workload in (:targeted, :full) || throw(ArgumentError("unknown workload"))
    expected = mask_kernel_reference(left, right, workload, target)
    return (; left, right, target, expected, n, representation, workload)
end

function mask_kernel_reference(left, right, workload, target)
    result = Dict{BigInt,Float64}()
    for (amask, avalue) in left, (bmask, bvalue) in right
        inversions = 0
        for i in 0:127
            iszero(amask & (one(amask) << i)) && continue
            for j in 0:i-1
                iszero(bmask & (one(bmask) << j)) || (inversions += 1)
            end
        end
        mask = BigInt(amask ⊻ bmask)
        result[mask] = get(result, mask, 0.0) +
                       (isodd(inversions) ? -1.0 : 1.0) * avalue * bvalue
    end
    return workload == :targeted ? get(result, BigInt(target), 0.0) :
        filter!(pair -> !iszero(last(pair)), result)
end

function mask_kernel_workload(state)
    (; left, right, target, workload) = state
    K = keytype(left)
    if workload == :targeted
        result = 0.0
        for (amask, avalue) in left
            bmask = amask ⊻ target
            bvalue = get(right, bmask, 0.0)
            iszero(bvalue) && continue
            result += Garamon._shuffle_sign_bits(amask, bmask) * avalue * bvalue
        end
        return result
    end
    result = Dict{K,Float64}()
    for (amask, avalue) in left, (bmask, bvalue) in right
        mask = amask ⊻ bmask
        value = get(result, mask, 0.0) +
                Garamon._shuffle_sign_bits(amask, bmask) * avalue * bvalue
        if iszero(value)
            delete!(result, mask)
        else
            result[mask] = value
        end
    end
    return result
end

function mask_kernel_oracle(state)
    result = mask_kernel_workload(state)
    if state.workload == :targeted
        return result == state.expected
    end
    return Dict(BigInt(k) => v for (k, v) in result) == state.expected
end
