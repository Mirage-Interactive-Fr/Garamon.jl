using Test
using Garamon

@testset "Prepared exact triple joins" begin
    for n in (2, 3, 4, 6, 8, 12, 64, 65, 66, 127, 128, 129)
        ga = algebra(n, :ega)
        a = scalar(ga, 2; storage=:sparse) +
            basisvector(ga, 1; storage=:sparse)
        b = scalar(ga, 3; storage=:sparse) +
            basisvector(ga, n; storage=:sparse) +
            basisblade(ga, [1, n]; storage=:sparse)
        c = scalar(ga, 4; storage=:sparse) +
            basisvector(ga, 1; storage=:sparse) +
            basisvector(ga, n; storage=:sparse) +
            basisblade(ga, [1, n]; storage=:sparse)
        outputs = [Int[], [1], [n], [1, n], Int[]]
        for (left, middle, right) in ((a, b, c), (c, a, b), (b, c, a))
            plan = prepare_triple_join(left, middle, right, outputs)
            @test length(plan.output_masks) == 4
            workspace = TripleJoinWorkspace(plan, left, middle, right)
            expected = (left * middle) * right
            values = run_triple_join_values!(workspace, left, middle, right)
            @test values == [coefficient_mask(expected, mask)
                             for mask in plan.output_masks]
            @test run_triple_join_values!(workspace, 2left, middle, right) === values
            @test values == [2coefficient_mask(expected, mask)
                             for mask in plan.output_masks]
            @test_throws ArgumentError TripleJoinWorkspace(plan, left, middle,
                                                            right; max_bytes=1)
            @test_throws ArgumentError run_triple_join_values!(workspace,
                basisvector(ga, 1; storage=:sparse), middle, right)
        end
    end

    ga = algebra(Rational{Int}[2 0 0; 0 0 0; 0 0 -3])
    a = basisvector(ga, 1; storage=:sparse) +
        basisvector(ga, 2; storage=:sparse)
    b = basisvector(ga, 2; storage=:sparse) +
        basisvector(ga, 3; storage=:sparse)
    c = scalar(ga, 2//1; storage=:sparse) + a
    plan = prepare_triple_join(a, b, c, [Int[], [1], [2], [3]])
    workspace = TripleJoinWorkspace(plan, a, b, c)
    @test run_triple_join_values!(workspace, a, b, c) ==
          [coefficient_mask((a * b) * c, mask) for mask in plan.output_masks]
    @test_throws ArgumentError prepare_triple_join(a, b, c, [Int[]]; max_probes=1)
    @test_throws ArgumentError prepare_triple_join(a, b, c, [[3]]; max_paths=0)
    @test_throws ArgumentError prepare_triple_join(a, b, c, [Int[]]; max_paths=-1)
    @test_throws ArgumentError TripleJoinWorkspace(plan, a, b, c; max_bytes=0)
    metric(ga)[1, 1] = 5//1
    @test_throws ArgumentError run_triple_join_values!(workspace, a, b, c)
    nonorthogonal = algebra(Rational{Int}[1 1; 1 1])
    u = basisvector(nonorthogonal, 1; storage=:sparse)
    @test_throws ArgumentError prepare_triple_join(u, u, u, [Int[]])
end
