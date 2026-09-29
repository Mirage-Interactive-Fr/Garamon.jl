using Garamon

function expression_grade_state()
    ga = algebra(10, :ega)
    masks = [UInt64(1) << (i - 1) | UInt64(1) << (j - 1)
             for i in 1:9 for j in i+1:10]
    a = multivector(ga, Dict(mask => Float64(mod(k, 5) + 1)
                             for (k, mask) in enumerate(masks)); storage=:sparse)
    b = multivector(ga, Dict(mask => Float64(mod(2k, 7) - 3)
                             for (k, mask) in enumerate(masks)
                             if mod(2k, 7) != 3); storage=:sparse)
    c = multivector(ga, Dict(mask => Float64(mod(3k, 11) - 5)
                             for (k, mask) in enumerate(masks)
                             if mod(3k, 11) != 5); storage=:sparse)
    plan = prepare_expression(@ga (a * b) * c)
    expected = scalarpart((a * b) * c)
    return (; plan, expected)
end

function expression_grade_workload(state, strategy::Symbol)
    output = Vector{Float64}(undef, 8)
    for i in eachindex(output)
        output[i] = strategy == :full ? scalarpart(evaluate(state.plan)) :
            evaluate(state.plan; output=Int[], strategy)
    end
    return output
end

expression_grade_oracle(state, strategy::Symbol) =
    all(value -> isapprox(value, state.expected; atol=1e-9, rtol=1e-9),
        expression_grade_workload(state, strategy))
