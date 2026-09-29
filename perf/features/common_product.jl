using Garamon

function product_state(n::Int; storage::Symbol=:sparse)
    ga = algebra(n, :ega)
    K = n <= 64 ? UInt64 : BigInt
    first_mask = n == 66 ? big(1) << 64 : zero(K)
    second_mask = n == 66 ? big(1) << 65 : convert(K, 8)
    a = multivector(ga, Dict(K(first_mask + i) => Float64(mod(i, 3) + 1)
                             for i in 0:7); storage)
    b = multivector(ga, Dict(K(second_mask + i) => Float64(mod(i, 5) + 1)
                             for i in 0:7); storage)
    expected = a * b
    return (; a, b, expected)
end

function product_workload(state, strategy::Symbol)
    plan = strategy == :direct ? nothing :
        prepare_product(state.a, state.b; max_paths=64)
    generated = strategy == :generated ? generate_product(plan; max_paths=64) : nothing
    output = Vector{AbstractMultiVector}(undef, 32)
    for i in eachindex(output)
        output[i] = strategy == :direct ? state.a * state.b :
                    strategy == :prepared ? run_product(plan, state.a, state.b) :
                    run_generated_product(generated, state.a, state.b)
    end
    return output
end

product_oracle(state, strategy::Symbol) =
    all(==(state.expected), product_workload(state, strategy))
