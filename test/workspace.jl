using Test
using Garamon
using LinearAlgebra

@testset "Reusable product workspaces" begin
    for n in (2, 3, 4, 5, 6, 8, 12, 66)
        ga = algebra(Matrix{Rational{Int}}(I, n, n))
        a = scalar(ga, 2//1; storage=:sparse) +
            basisvector(ga, 1; storage=:sparse) +
            basisblade(ga, [1, n]; storage=:sparse)
        b = scalar(ga, 3//1; storage=:sparse) +
            basisvector(ga, n; storage=:sparse)
        plan = prepare_product(a, b)
        workspace = ProductWorkspace(plan, a, b)
        expected = a * b
        values = run_product_values!(workspace, a, b)
        @test values == [coefficient_mask(expected, mask)
                         for mask in plan.output_masks]
        result = run_product!(workspace, a, b)
        @test result == expected
        saved = sparse(result)
        changed = 2 * a
        @test run_product!(workspace, changed, b) === result
        @test result == changed * b
        @test saved == expected
        @test_throws ArgumentError ProductWorkspace(plan, a, b; max_bytes=1)
        @test_throws ArgumentError run_product_values!(
            workspace, a + basisvector(ga, min(2, n); storage=:sparse), b)
    end

    degenerate = algebra(Rational{Int}[1 0; 0 0])
    a = scalar(degenerate, 1//1; storage=:sparse) +
        basisvector(degenerate, 2; storage=:sparse)
    plan = prepare_product(a, a)
    workspace = ProductWorkspace(plan, a, a)
    @test run_product!(workspace, a, a) == a * a
    @test_throws ArgumentError ProductWorkspace(plan, a, a; max_bytes=0)
end

@testset "Fixed-width blade masks across machine-word boundaries" begin
    for (n, expected_type) in ((63, UInt64), (64, UInt64),
                               (65, UInt128), (70, UInt128),
                               (127, UInt128), (128, UInt128),
                               (129, BigInt))
        ga = algebra(n, :ega)
        a = basisvector(ga, 1; storage=:sparse) +
            basisvector(ga, n; storage=:sparse)
        b = basisvector(ga, n; storage=:sparse)
        @test keytype(a.values) == expected_type
        @test coefficient(a * b, Int[]) == 1
        @test coefficient(a * b, [1, n]) == 1
        mask = blade_unrank(ga, 2, binomial(big(n), 2) - 1)
        @test mask isa expected_type
        @test blade_rank(ga, mask) == (2, binomial(big(n), 2) - 1)
        @test right_uncomplement(right_complement(a)) == a
        @test_throws BoundsError coefficient_mask(a, big(1) << n)
        @test_throws BoundsError multivector(ga,
            Dict((big(1) << n) => 1); storage=:sparse)
        plan = prepare_product(a, b)
        workspace = ProductWorkspace(plan, a, b)
        @test run_product!(workspace, a, b) == a * b
    end
end
