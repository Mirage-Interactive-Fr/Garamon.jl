include(joinpath(@__DIR__, "common_product.jl"))

function coefficient_workload(state, strategy::Symbol)
    output = Vector{Float64}(undef, 32)
    target = [65, 66]
    mask = (big(1) << 64) | (big(1) << 65)
    for i in eachindex(output)
        output[i] = strategy == :targeted ?
            product_coefficient(state.a, state.b, target) :
            coefficient_mask(state.a * state.b, mask)
    end
    return output
end

coefficient_oracle(state, strategy::Symbol) =
    all(==(coefficient_mask(state.expected,
                           (big(1) << 64) | (big(1) << 65))),
        coefficient_workload(state, strategy))
