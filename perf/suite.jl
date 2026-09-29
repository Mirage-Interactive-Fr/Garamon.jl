using PerfChecker

function build_suite()
    features = FeatureSpec[]
    for trace in (:hot, :scan, :phase, :shift), policy in (:direct, :lru, :roulette)
        name = Symbol(trace, :_, policy)
        push!(features, FeatureSpec(name;
            description = "Exact geometric products, $(trace) cache trace, $(policy) policy",
            backend = :benchmark,
            entrypoint = joinpath(@__DIR__, "features", "$(name).jl"),
            comparison_key = "garamon/cache/$(trace)/v1",
            oracle = OracleSpec(function_name = :perf_oracle),
            options = Dict(:tags => [:garamon, :cache, trace, policy],
                           :samples => 25, :evals => 1, :seconds => 0.5)))
    end
    for dimension in (4, 12, 66), strategy in (:direct, :prepared, :generated)
        name = Symbol(:product_, dimension, :_, strategy)
        push!(features, FeatureSpec(name;
            description = "32 exact products in $(dimension)D with $(strategy)",
            backend = :benchmark,
            entrypoint = joinpath(@__DIR__, "features", "$(name).jl"),
            comparison_key = "garamon/product/$(dimension)d/reuse32/v1",
            oracle = OracleSpec(function_name = :perf_oracle),
            options = Dict(:tags => [:garamon, :product, strategy],
                           :samples => 25, :evals => 1, :seconds => 0.5)))
    end
    for strategy in (:direct, :prepared, :generated)
        name = Symbol(:product_4_dense_, strategy)
        push!(features, FeatureSpec(name;
            description = "32 exact dense-storage products in 4D with $(strategy)",
            backend = :benchmark,
            entrypoint = joinpath(@__DIR__, "features", "$(name).jl"),
            comparison_key = "garamon/product/4d-dense/reuse32/v1",
            oracle = OracleSpec(function_name = :perf_oracle),
            options = Dict(:tags => [:garamon, :product, :dense, strategy],
                           :samples => 25, :evals => 1, :seconds => 0.5)))
    end
    for strategy in (:targeted, :full)
        name = Symbol(:coefficient_66_, strategy)
        push!(features, FeatureSpec(name;
            description = "32 requests for one exact 66D product coefficient",
            backend = :benchmark,
            entrypoint = joinpath(@__DIR__, "features", "$(name).jl"),
            comparison_key = "garamon/coefficient/66d/reuse32/v1",
            oracle = OracleSpec(function_name = :perf_oracle),
            options = Dict(:tags => [:garamon, :coefficient, strategy],
                           :samples => 25, :evals => 1, :seconds => 0.5)))
    end
    for strategy in (:full, :recursive, :grades, :join3)
        name = Symbol(:expression_scalar_, strategy)
        push!(features, FeatureSpec(name;
            description = "Eight exact scalar requests from a nested 10D grade-2 expression",
            backend = :benchmark,
            entrypoint = joinpath(@__DIR__, "features", "$(name).jl"),
            comparison_key = "garamon/expression/10d/scalar8/v1",
            oracle = OracleSpec(function_name = :perf_oracle),
            options = Dict(:tags => [:garamon, :expression, strategy],
                           :samples => 25, :evals => 1, :seconds => 0.5)))
    end
    package = PackageSuite("Garamon";
        worker_environment = joinpath(@__DIR__, "runner"),
        source = dirname(@__DIR__), versions = VersionNumber[],
        dev_sources = String[], features)
    return SoftwareSuite(:garamon_cache_policies, [package];
        description = "Direct products, LRU, and exact-product roulette cache")
end
