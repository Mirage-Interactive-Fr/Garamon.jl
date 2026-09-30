@testset "Multivector products and storage" begin
    metric_exact = Rational{Int}[2 1 0; 1 3 0; 0 0 0]
    ga = algebra(metric_exact)
    e = [basisvector(ga, i; storage=:sparse) for i in 1:3]
    one_ga = scalar(ga, 1//1; storage=:sparse)

    @test scalarpart(e[1] * e[2]) == 1//1
    @test coefficient(e[1] * e[2], [1, 2]) == 1//1
    @test e[1] * e[2] + e[2] * e[1] == 2 * one_ga
    @test e[3] * e[3] == zero(e[3])
    @test e[1] ∧ e[1] == zero(e[1])
    @test e[1] ∧ e[2] == -(e[2] ∧ e[1])
    @test left_contraction(e[1], e[2]) == one_ga
    @test right_contraction(e[1] ∧ e[2], e[2]) ==
          grade((e[1] ∧ e[2]) * e[2], 1)

    dense_values = zeros(Rational{Int}, 8)
    owned_dense = DenseMultiVector(ga, dense_values)
    dense_values[1] = 3//1
    @test scalarpart(owned_dense) == 3//1
    sparse_values = Dict(UInt64(0) => 3//1)
    copied_sparse = SparseMultiVector(ga, sparse_values)
    sparse_values[UInt64(0)] = 4//1
    @test scalarpart(copied_sparse) == 3//1

    blades = [multivector(ga, Dict(UInt64(mask) => 1//1); storage=:sparse)
              for mask in 0:7]
    for a in blades, b in blades, c in blades
        @test (a * b) * c == a * (b * c)
        @test reverse(a * b) == reverse(b) * reverse(a)
    end

    for a in blades, b in blades
        @test dense(a) * dense(b) == a * b
        @test sparse(dense(a)) == a
        for operation in (:geometric, :wedge, :left, :right), mask in 0:7
            product = operation == :geometric ? a * b :
                      operation == :wedge ? wedge(a, b) :
                      operation == :left ? left_contraction(a, b) :
                      right_contraction(a, b)
            @test product_coefficient(a, b, [i for i in 1:3 if mask & (1 << (i - 1)) != 0]; operation) ==
                  coefficient_mask(product, mask)
        end
    end
    @test_throws ArgumentError basisblade(ga, [1, 1])
    @test_throws BoundsError basisvector(ga, 4)
    @test_throws ArgumentError algebra([1 2; 0 1])
    @test_throws DimensionMismatch algebra([1 0; 0 1]; basis=["one"])
    @test_throws ArgumentError geometric_product(e[1], e[2]; max_terms=1)
    @test_throws ArgumentError geometric_product(e[1], e[2]; max_terms=0)
    requests = (Int[i for i in 1:3 if mask & (1 << (i - 1)) != 0]
                for mask in 0:7)
    left, right = e[1] + e[2], e[2] + e[3]
    for operation in (:geometric, :wedge, :left, :right)
        product = operation == :geometric ? left * right :
                  operation == :wedge ? wedge(left, right) :
                  operation == :left ? left_contraction(left, right) :
                  right_contraction(left, right)
        expected = [coefficient_mask(product, mask) for mask in 0:7]
        for strategy in (:full, :auto)
            @test product_coefficients(left, right, requests;
                                       operation, strategy) == expected
        end
    end
    diagonal_ga = algebra(Rational{Int}[1 0 0; 0 2 0; 0 0 3])
    da, db, _ = basisvectors(diagonal_ga; storage=:sparse)
    for operation in (:geometric, :wedge, :left, :right)
        @test product_coefficients(da + db, db + da, requests;
                                   operation, strategy=:targeted) ==
              product_coefficients(da + db, db + da, requests;
                                   operation, strategy=:full)
        @test product_coefficients(da + db, db, requests;
                                   operation, strategy=:targeted) ==
              product_coefficients(da + db, db, requests;
                                   operation, strategy=:full)
    end
    @test product_coefficients(left, right, [Int[], Int[]]) ==
          [scalarpart(left * right), scalarpart(left * right)]
    @test isempty(product_coefficients(left, right, Int[]))
    @test_throws ArgumentError product_coefficients(left, right, [Int[]];
                                                    strategy=:unknown)
    @test_throws ArgumentError product_coefficients(e[1], e[2], [Int[]];
                                                    strategy=:targeted)
end

using LinearAlgebra

@testset "Exterior maps and factorized blades" begin
    ga = algebra(Rational{Int}[1 0 0; 0 1 0; 0 0 1])
    P = Rational{Int}[1 1 0; 0 1 1; 0 0 1]
    a = basisvector(ga, 1) + 2 * basisvector(ga, 2)
    b = basisvector(ga, 2) - basisvector(ga, 3)
    @test outermorphism(P, a ∧ b, ga) ==
          outermorphism(P, a, ga) ∧ outermorphism(P, b, ga)
    @test outermorphism(inv(P), outermorphism(P, a ∧ b, ga), ga) == a ∧ b
    @test_throws ArgumentError outermorphism(P, a, ga; check_metric=true)
    @test outermorphism(Matrix{Rational{Int}}(I, 3, 3), a, ga; check_metric=true) == a
    source = algebra(Rational{Int}[2 0; 0 3])
    target = algebra(Rational{Int}[3 0; 0 2])
    swap = Rational{Int}[0 1; 1 0]
    sa, sb = basisvectors(source)
    @test outermorphism(swap, sa * sb, target; check_metric=true) ==
          outermorphism(swap, sa, target; check_metric=true) *
          outermorphism(swap, sb, target; check_metric=true)
    @test_throws DimensionMismatch outermorphism(ones(2, 2), a, ga)
    plane = algebra(Rational{Int}[1 0; 0 1])
    embedding = Rational{Int}[1 0; 0 1; 0 0]
    @test outermorphism(embedding, scalar(plane, 3//1), ga) == scalar(ga, 3//1)
    @test outermorphism(embedding, basisblade(plane, [1, 2]), ga) ==
          basisblade(ga, [1, 2])
    singular = Rational{Int}[1 1; 0 0; 0 0]
    @test outermorphism(singular, basisblade(plane, [1, 2]), ga) ==
          zero(basisvector(ga, 1; storage=:sparse))
    complex_ga = algebra(ComplexF64[1;;])
    complex_vector = basisvector(complex_ga, 1)
    @test_throws ArgumentError outermorphism(reshape([im], 1, 1), complex_vector,
                                            complex_ga; check_metric=true)

    factors = Rational{Int}[1 0; 2 1; 0 1]
    f = FactorizedBlade(ga, factors; scale=2//1)
    @test coefficient(f, [1, 2]) == 2//1
    @test coefficient(f, [1, 3]) == 2//1
    @test coefficient(f, [2, 3]) == 4//1
    @test expand(f) == 2 * ((basisvector(ga, 1) + 2 * basisvector(ga, 2)) ∧
                            (basisvector(ga, 2) + basisvector(ga, 3)))
    @test expand(outermorphism(P, f, ga)) == outermorphism(P, expand(f), ga)
    @test_throws ArgumentError expand(f; max_terms=1)
    integer_ga = algebra([1 0; 0 1])
    q = 1 << 30
    large_integer_blade = FactorizedBlade(integer_ga, [q + 1 q; q q - 1])
    @test coefficient(large_integer_blade, [1, 2]) == -1
    @test coefficient(large_integer_blade, [1, 2]) ==
          coefficient(expand(large_integer_blade), [1, 2])

    u = [1//1, 1//1, 0//1]
    v = [2//1, 0//1, 1//1]
    chain = ReflectionChain(ga, reshape(u, 3, 1))
    transformed = versor_action(chain, v)
    uv = sum(v[i] * basisvector(ga, i) for i in 1:3)
    uu = sum(u[i] * basisvector(ga, i) for i in 1:3)
    @test transformed == [coefficient(-uu * uv * inv(uu), [i]) for i in 1:3]
    @test expand(versor_action(chain, f)) ==
          outermorphism(hcat([versor_action(chain, [i == j ? 1//1 : 0//1 for i in 1:3])
                               for j in 1:3]...), expand(f), ga)
    @test_throws DomainError ReflectionChain(ga, zeros(Rational{Int}, 3, 1))
    indefinite = algebra(Rational{Int}[2 1 0; 1 -1 0; 0 0 1])
    normals = Rational{Int}[1 0; 0 1; 0 0]
    sequence = ReflectionChain(indefinite, normals)
    vector = Rational{Int}[2, 3, 4]
    u1, u2 = basisvector(indefinite, 1), basisvector(indefinite, 2)
    v = sum(vector[i] * basisvector(indefinite, i) for i in 1:3)
    expected = u2 * u1 * v * inv(u1) * inv(u2)
    @test versor_action(sequence, vector) ==
          [coefficient(expected, [i]) for i in 1:3]

    large = algebra(70, :ega)
    unit_factors = zeros(Float64, 70, 2)
    unit_factors[1, 1] = unit_factors[70, 2] = 1.0
    large_blade = FactorizedBlade(large, unit_factors)
    @test coefficient(large_blade, [1, 70]) == 1.0
    @test expand(large_blade) == basisblade(large, [1, 70])
    large_chain = ReflectionChain(large, unit_factors[:, 1:1])
    @test versor_action(large_chain, unit_factors[:, 1]) == -unit_factors[:, 1]
end

@testset "Metric congruence and coordinate directions" begin
    source = algebra(Rational{Int}[0 1 0; 1 0 0; 0 0 0])
    decomposition = diagonalize_metric(source)
    @test decomposition.rank == 2
    @test isdiag(metric(decomposition.orthogonal))
    @test transpose(decomposition.forward) * metric(source) *
          decomposition.forward == metric(decomposition.orthogonal)
    a = basisvector(source, 1) + 2 * basisvector(source, 2)
    b = basisvector(source, 2) + basisvector(source, 3)
    oa, ob = to_orthogonal(decomposition, a), to_orthogonal(decomposition, b)
    @test from_orthogonal(decomposition, oa, source) == a
    @test from_orthogonal(decomposition, oa * ob, source) == a * b
    @test_throws ArgumentError diagonalize_metric(source; max_dimension=2)
    @test_throws ArgumentError to_orthogonal(decomposition, basisvector(algebra(3, :ega), 1))

    numerical = algebra([2.0 1.0; 1.0 3.0])
    data = diagonalize_metric(numerical)
    x = basisvector(numerical, 1) + basisvector(numerical, 2)
    @test isapprox(dense(from_orthogonal(data, to_orthogonal(data, x), numerical)).values,
                   dense(x).values; atol=1e-12)
    conformal = algebra(3, :cga)
    conformal_data = diagonalize_metric(conformal)
    @test conformal_data.rank == 5
    @test isdiag(metric(conformal_data.orthogonal))
    origin, infinity = basisvector(conformal, 1), basisvector(conformal, 5)
    converted_product = from_orthogonal(conformal_data,
        to_orthogonal(conformal_data, origin) *
        to_orthogonal(conformal_data, infinity), conformal)
    @test isapprox(dense(converted_product).values,
                   dense(origin * infinity).values; atol=1e-12)
    tiny_metric = algebra([1.0 0.0; 0.0 1e-10])
    @test_throws ArgumentError diagonalize_metric(tiny_metric)
    @test metric(diagonalize_metric(tiny_metric; rtol=0).orthogonal)[2, 2] == 1e-10
end

@testset "Active coordinate subspaces" begin
    ga = algebra(66, :ega)
    a = basisvector(ga, 1; storage=:sparse) +
        2basisvector(ga, 66; storage=:sparse)
    b = 3basisvector(ga, 66; storage=:sparse) +
        scalar(ga, 1; storage=:sparse)
    plan = coordinate_subspace(a, b)
    @test plan.indices == [1, 66]
    @test dimension(plan.algebra) == 2
    @test lift_subspace(plan, project_subspace(plan, a), ga) == a
    for operation in (:geometric, :wedge, :left, :right)
        reference = operation == :geometric ? a * b :
                    operation == :wedge ? wedge(a, b) :
                    operation == :left ? left_contraction(a, b) :
                    right_contraction(a, b)
        @test subspace_product(plan, a, b; operation) == reference
    end
    @test_throws ArgumentError project_subspace(plan,
                                                basisvector(ga, 2; storage=:sparse))
    saved = basis(ga)[1]
    basis(ga)[1] = "changed"
    @test_throws ArgumentError project_subspace(plan, a)
    basis(ga)[1] = saved

    offdiagonal = algebra(Rational{Int}[2 1 0; 1 3 1; 0 1 4])
    u = basisvector(offdiagonal, 1; storage=:sparse) +
        basisvector(offdiagonal, 3; storage=:sparse)
    v = basisvector(offdiagonal, 3; storage=:sparse)
    restricted = coordinate_subspace(u, v)
    @test restricted.indices == [1, 3]
    @test subspace_product(restricted, u, v) == u * v
    metric(offdiagonal)[1, 3] = metric(offdiagonal)[3, 1] = 1//2
    @test_throws ArgumentError subspace_product(restricted, u, v)
end

@testset "Inverse and grade semantics" begin
    ga = algebra(Rational{Int}[1;;])
    e1 = basisvector(ga, 1)
    a = 2 * scalar(ga, 1//1) + e1
    @test inv(a) == (2 * scalar(ga, 1//1) - e1) / 3
    @test a * inv(a) == scalar(ga, 1//1)
    @test_throws DomainError inv(scalar(ga, 1//1) + e1)

    weighted = algebra(Rational{Int}[2 0; 0 3])
    Iblade = basisblade(weighted, [1, 2])
    @test Iblade * Iblade == scalar(weighted, -6//1)
    @test inv(Iblade) == -Iblade / 6

    four = algebra(4, :ega)
    bivector = basisblade(four, [1, 2]) + basisblade(four, [3, 4])
    @test coefficient(bivector ∧ bivector, [1, 2, 3, 4]) == 2

    compound = scalar(weighted, 2//1) + basisvector(weighted, 1) + Iblade
    @test grade_involution(compound) ==
          scalar(weighted, 2//1) - basisvector(weighted, 1) + Iblade
    @test clifford_conjugate(compound) ==
          scalar(weighted, 2//1) - basisvector(weighted, 1) - Iblade
    @test metric_undual(metric_dual(compound)) == compound
    @test right_uncomplement(right_complement(compound)) == compound
    @test basisvector(weighted, 1) ∧ right_complement(basisvector(weighted, 1)) == Iblade
    degenerate = algebra(Rational{Int}[0 0; 0 1])
    @test right_uncomplement(right_complement(basisvector(degenerate, 1))) ==
          basisvector(degenerate, 1)
    @test_throws DomainError metric_dual(basisvector(degenerate, 1))
    skewed = algebra(Rational{Int}[2 1; 1 3])
    skewed_value = scalar(skewed, 3//1) + basisvector(skewed, 1) +
                   basisblade(skewed, [1, 2])
    @test metric_undual(metric_dual(skewed_value)) == skewed_value

    complex_ga = algebra(ComplexF64[1;;])
    complex_a = scalar(complex_ga, 1.2 + 0.7im) +
                (0.4 + 0.3im) * basisvector(complex_ga, 1)
    complex_inverse = inv(complex_a)
    @test isapprox(dense(complex_a * complex_inverse).values,
                   dense(scalar(complex_ga, 1)).values; atol=1e-12)
    @test isapprox(dense(complex_inverse * complex_a).values,
                   dense(scalar(complex_ga, 1)).values; atol=1e-12)
end

@testset "C++ runtime operation coverage" begin
    ga = algebra(Rational{Int}[2 0 0; 0 -3 0; 0 0 0])
    e = basisvectors(ga; storage=:sparse)
    a = scalar(ga, 2//1; storage=:sparse) + e[1] +
        basisblade(ga, [1, 2]; storage=:sparse)
    b = scalar(ga, 3//1; storage=:sparse) - e[2] +
        basisblade(ga, [2, 3]; storage=:sparse)
    parts_a = [grade(a, k) for k in 0:3]
    parts_b = [grade(b, k) for k in 0:3]
    expected_dot = zero(a)
    expected_inner = zero(a)
    expected_scalar = zero(a)
    for r in 0:3, s in 0:3
        piece = grade(parts_a[r + 1] * parts_b[s + 1], abs(r - s))
        expected_dot += piece
        r > 0 && s > 0 && (expected_inner += piece)
        r == s && (expected_scalar += grade(piece, 0))
    end
    @test dot_product(a, b) == expected_dot
    @test inner_product(a, b) == expected_inner
    @test scalar_product(a, b) == expected_scalar
    @test scalar_product(a, b) == grade(a * b, 0)
    @test iszero(inner_product(scalar(ga, 2//1), e[1]))
    @test dot_product(scalar(ga, 2//1), e[1]) == 2 * e[1]
    @test quadratic_norm(e[2]) == -3//1
    @test clifford_norm(e[2]) == sqrt(3)

    for operation in (:inner, :dot, :scalar)
        direct = operation == :inner ? inner_product(a, b) :
                 operation == :dot ? dot_product(a, b) : scalar_product(a, b)
        plan = prepare_product(a, b; operation)
        @test run_product(plan, a, b) == direct
        for mask in 0:7
            indices = [i for i in 1:3 if mask & (1 << (i - 1)) != 0]
            @test product_coefficient(a, b, indices; operation) ==
                  coefficient(direct, indices)
        end
    end

    regular = algebra(Rational{Int}[1 0; 0 1])
    x = scalar(regular, 2//1) + basisvector(regular, 1)
    @test x + 3 == 3 + x == scalar(regular, 5//1) + basisvector(regular, 1)
    @test 3 - x == scalar(regular, 1//1) - basisvector(regular, 1)
    @test x - 3 == x + (-3)
    @test x / x == scalar(regular, 1//1)
    @test 3 / x == 3 * inv(x)
    @test wedge(x, 3) == 3x == wedge(3, x)
    @test iszero(inner_product(x, 3))
    @test iszero(inner_product(3, x))
    @test dot_product(x, 3) == 3x == dot_product(3, x)
    @test left_contraction(3, x) == 3x
    @test right_contraction(x, 3) == 3x
    @test left_contraction(x, 3) == scalar(regular, 6//1)
    @test right_contraction(3, x) == scalar(regular, 6//1)
    @test scalar_product(x, 3) == scalar(regular, 6//1)
    @test active_grades(a) == [0, 1, 2]
    @test highest_grade(a) == 2
    @test is_homogeneous(grade(a, 2))
    @test has_grade(a, 2)
    @test has_grade(zero(a), 0)
    @test same_grade(a, grade(a, 2))
    @test iszero(zero(a))
    @test !iszero(a)
    mutable_a = copy(a)
    set_coefficient!(mutable_a, [3], 5//1)
    @test coefficient(mutable_a, [3]) == 5//1
    @test coefficient(a, [3]) == 0
    @test coefficient_grade(a, 1, 0) == coefficient(a, [1])
    set_grade_coefficient!(mutable_a, 1, 1, 7//1)
    @test coefficient(mutable_a, [2]) == 7//1
    clear_grade!(mutable_a, 2)
    @test active_grades(mutable_a) == [0, 1]
    set_coefficient!(mutable_a, [3], 1//100)
    round_zero!(mutable_a; atol=1//50)
    @test coefficient(mutable_a, [3]) == 0
    @test_throws ArgumentError round_zero!(mutable_a; atol=-1)
    empty!(mutable_a)
    @test iszero(mutable_a)
    @test outer_primal_dual(e[1], e[2]) ==
          wedge(e[1], right_complement(e[2]))
    @test outer_dual_primal(e[1], e[2]) ==
          wedge(right_complement(e[1]), e[2])
    @test outer_dual_dual(e[1], e[2]) ==
          wedge(right_complement(e[1]), right_complement(e[2]))
end

@testset "Homogeneous blade indexing without a global table" begin
    ga = algebra(4, :ega)
    expected = UInt64[3, 5, 9, 6, 10, 12]
    for (position, mask) in enumerate(expected)
        @test blade_rank(ga, mask) == (2, big(position - 1))
        @test blade_unrank(ga, 2, position - 1) == mask
    end
    for mask in 0:15
        grade, position = blade_rank(ga, mask)
        @test blade_unrank(ga, grade, position) == mask
    end
    @test blade_rank(ga, [1, 3]) == (2, big(1))
    @test blade_rank(ga, 3) == (2, big(0))
    @test blade_unrank(ga, 0, 0) == 0
    @test_throws BoundsError blade_unrank(ga, 2, 6)
    @test_throws BoundsError blade_rank(ga, 16)
    high = algebra(70, :ega)
    mask = (big(1) << 69) | (big(1) << 2)
    grade, position = blade_rank(high, mask)
    @test grade == 2
    @test blade_unrank(high, grade, position) == mask
end

@testset "Bounded on-demand Julia product generation" begin
    ga = algebra(Rational{Int}[2 0; 0 -3])
    a = basisvector(ga, 1; storage=:sparse) +
        basisvector(ga, 2; storage=:sparse)
    b = scalar(ga, 2//1; storage=:sparse) +
        basisvector(ga, 1; storage=:sparse)
    for operation in (:geometric, :wedge, :left, :right, :inner, :dot, :scalar)
        plan = prepare_product(a, b; operation)
        generated = generate_product(plan; max_paths=4)
        @test run_generated_product(generated, a, b) == run_product(plan, a, b)
        @test run_generated_product(generated, 3a, 2b) ==
              6 * run_product(plan, a, b)
        @test_throws ArgumentError run_generated_product(generated,
            basisvector(ga, 1; storage=:sparse), b)
    end
    plan = prepare_product(a, b)
    @test_throws ArgumentError generate_product(plan; max_paths=3)
    @test_throws ArgumentError generate_product(plan; max_paths=257)
    mutable_metric = algebra([1 0; 0 1])
    u = basisvector(mutable_metric, 1; storage=:sparse)
    generated = generate_product(prepare_product(u, u))
    metric(mutable_metric)[1, 1] = 2
    @test_throws ArgumentError run_generated_product(generated, u, u)

    # Different absolute masks can share the same generated path topology.
    ga8 = algebra(8, :ega)
    directions = basisvectors(ga8; storage=:sparse)
    first_program = generate_product(prepare_product(
        directions[1] + directions[2], directions[3] + directions[4]))
    second_program = generate_product(prepare_product(
        directions[5] + directions[6], directions[7] + directions[8]))
    @test typeof(first_program) === typeof(second_program)
    @test run_generated_product(first_program,
        directions[1] + directions[2], directions[3] + directions[4]) ==
        (directions[1] + directions[2]) * (directions[3] + directions[4])
    @test run_generated_product(second_program,
        directions[5] + directions[6], directions[7] + directions[8]) ==
        (directions[5] + directions[6]) * (directions[7] + directions[8])
end

@testset "Active support beyond machine-word masks" begin
    ga = algebra(66, :ega)
    a = basisvector(ga, 1)
    b = basisvector(ga, 66)
    @test a isa SparseMultiVector
    @test coefficient(a * b, [1, 66]) == 1
    @test coefficient(b * a, [1, 66]) == -1
    @test length((a * b).values) == 1
    @test product_coefficient(a, b, [1, 66]) == 1
    @test product_coefficient(b, a, [1, 66]) == -1
    @test right_uncomplement(right_complement(a)) == a
    for n in (63, 64, 65)
        boundary = algebra(n, :ega)
        edge = basisvector(boundary, n; storage=:sparse)
        @test right_uncomplement(right_complement(edge)) == edge
        @test metric_undual(metric_dual(edge)) == edge
    end
    for n in (128, 129)
        boundary = algebra(n, :ega)
        first_blade = basisvector(boundary, 1; storage=:sparse)
        last_blade = basisvector(boundary, n; storage=:sparse)
        @test first_blade * last_blade == wedge(first_blade, last_blade)
        @test last_blade * first_blade == -(first_blade * last_blade)
        @test last_blade * last_blade == scalar(boundary, 1; storage=:sparse)
        @test typeof(first(keys((first_blade * last_blade).values))) ==
              (n == 128 ? UInt128 : BigInt)
    end
    @test_throws ArgumentError dense(a)
end

@testset "Prepared products and homogeneous batches" begin
    ga = algebra(Rational{Int}[2 0 0; 0 0 0; 0 0 -1])
    a = basisvector(ga, 1; storage=:sparse) +
        2 * basisblade(ga, [2, 3]; storage=:sparse)
    b = basisvector(ga, 3; storage=:sparse) +
        3 * basisblade(ga, [1, 2]; storage=:sparse)
    for operation in (:geometric, :wedge, :left, :right)
        plan = prepare_product(a, b; operation)
        expected = operation == :geometric ? a * b :
                   operation == :wedge ? wedge(a, b) :
                   operation == :left ? left_contraction(a, b) :
                   right_contraction(a, b)
        @test run_product(plan, a, b) == expected
        @test batch_product(plan, [a, 2a], [b, 3b]) == [expected, 6expected]
        packed = pack_product_batch(plan, [a, 2a], [b, 3b])
        packed_values = run_packed_batch(packed)
        @test size(packed_values) == (length(plan.output_masks), 2)
        @test unpack_product_batch(packed, packed_values) ==
              [expected, 6expected]
        @test_throws DimensionMismatch unpack_product_batch(packed,
                                                            zeros(Int, 1, 1))
    end
    plan = prepare_product(a, b)
    @test_throws ArgumentError run_product(plan, basisvector(ga, 1), b)
    @test_throws ArgumentError pack_product_batch(plan,
        [basisvector(ga, 1)], [b])
    @test_throws ArgumentError prepare_product(a, b; max_paths=3)
    @test_throws DimensionMismatch batch_product(plan, [a], [b, b])
    @test_throws DimensionMismatch pack_product_batch(plan, [a], [b, b])
    packed = pack_product_batch(plan, [a], [b])
    equivalent_ga = algebra(copy(metric(ga)); basis=copy(basis(ga)))
    equivalent_a = basisvector(equivalent_ga, 1; storage=:sparse) +
        2 * basisblade(equivalent_ga, [2, 3]; storage=:sparse)
    equivalent_b = basisvector(equivalent_ga, 3; storage=:sparse) +
        3 * basisblade(equivalent_ga, [1, 2]; storage=:sparse)
    mixed_algebras = pack_product_batch(plan, [a, equivalent_a], [b, equivalent_b])
    @test run_packed_batch(mixed_algebras)[:, 1] ==
          run_packed_batch(mixed_algebras)[:, 2]
    saved_metric = metric(ga)[1, 1]
    metric(ga)[1, 1] = 7//1
    @test_throws ArgumentError run_product(plan, a, b)
    @test_throws ArgumentError run_packed_batch(packed)
    @test_throws ArgumentError pack_product_batch(plan, [a, 2a], [b, 3b])
    metric(ga)[1, 1] = saved_metric
    saved_name = basis(ga)[1]
    basis(ga)[1] = "changed"
    @test_throws ArgumentError run_product(plan, a, b)
    @test_throws ArgumentError pack_product_batch(plan, [a, 2a], [b, 3b])
    basis(ga)[1] = saved_name
    @test run_product(plan, a, b) == a * b
    offdiagonal = algebra(Rational{Int}[1 1; 1 1])
    @test_throws ArgumentError prepare_product(basisvector(offdiagonal, 1),
                                               basisvector(offdiagonal, 2))
end

@testset "Bounded homogeneous grade products" begin
    ga = algebra(Rational{Int}[2 0 0 0; 0 -3 0 0;
                                0 0 0 0; 0 0 0 5])
    for rgrade in 0:4, sgrade in 0:4,
        operation in (:geometric, :wedge, :left, :right)
        left_masks = [UInt64(mask) for mask in 0:15
                      if count_ones(mask) == rgrade]
        right_masks = [UInt64(mask) for mask in 0:15
                       if count_ones(mask) == sgrade]
        a = multivector(ga,
            Dict(mask => (i % 3 + 1)//1 for (i, mask) in
                 enumerate(left_masks) if isodd(i)); storage=:sparse)
        b = multivector(ga,
            Dict(mask => (i % 4 + 1)//1 for (i, mask) in
                 enumerate(right_masks) if isodd(i)); storage=:sparse)
        plan = prepare_grade_product(ga, rgrade, sgrade; operation)
        expected = operation == :geometric ? a * b :
                   operation == :wedge ? wedge(a, b) :
                   operation == :left ? left_contraction(a, b) :
                   right_contraction(a, b)
        @test run_grade_product(plan, a, b) == expected
    end
    grade_plan = prepare_grade_product(ga, 2, 2)
    a = basisblade(ga, [1, 2]; storage=:sparse)
    b = basisblade(ga, [3, 4]; storage=:sparse)
    @test_throws ArgumentError run_product(grade_plan, a, b)
    @test run_grade_product(grade_plan, a, b) == a * b
    @test_throws ArgumentError run_grade_product(grade_plan,
        a + basisvector(ga, 1; storage=:sparse), b)
    @test_throws ArgumentError prepare_grade_product(ga, 2, 2; max_paths=35)
    @test_throws ArgumentError prepare_grade_product(ga, 2, 2; max_slots=5)
    @test_throws ArgumentError prepare_grade_product(ga, -1, 1)
    @test_throws ArgumentError prepare_grade_product(
        algebra(Rational{Int}[2 1; 1 3]), 1, 1)
    large = algebra(66, :ega)
    large_plan = prepare_grade_product(large, 1, 1; max_paths=4356)
    @test run_grade_product(large_plan,
        basisvector(large, 66; storage=:sparse),
        basisvector(large, 1; storage=:sparse)) ==
        basisvector(large, 66; storage=:sparse) *
        basisvector(large, 1; storage=:sparse)
end

@testset "Bounded structural plan cache" begin
    ga = algebra(Rational{Int}[2 0 0; 0 3 0; 0 0 4])
    a, b, c = basisvectors(ga; storage=:sparse)
    cache = ProductPlanCache(max_bytes=10_000)
    @test cached_product!(cache, a, b) == a * b
    @test cached_product!(cache, 2a, 3b) == 6(a * b)
    @test cache_stats(cache).hits == 1
    @test cache_stats(cache).misses == 1
    @test cache_stats(cache).estimated_bytes <= cache_stats(cache).max_bytes
    saved = metric(ga)[1, 1]
    metric(ga)[1, 1] = 7//1
    @test cached_product!(cache, a, a) == a * a
    @test cache_stats(cache).misses == 2
    metric(ga)[1, 1] = saved
    @test cached_product!(cache, a, b) == a * b
    @test cache_stats(cache).hits == 2
    @test cached_product!(cache, a, c) == a * c
    @test cache_stats(cache).misses == 3
    oldname = basis(ga)[1]
    basis(ga)[1] = "other"
    @test cached_product!(cache, a, b) == a * b
    @test cache_stats(cache).misses == 4
    basis(ga)[1] = oldname
    @test_throws ArgumentError cached_product!(cache, a, b; max_paths=0)
    for n in (3, 8, 65, 129)
        sized_ga=algebra(n,:ega)
        left=basisvector(sized_ga,1;storage=:sparse)
        right=basisvector(sized_ga,n;storage=:sparse)
        key=Garamon._plan_cache_key(left,right,:geometric,1 << 16)
        plan=prepare_product(left,right)
        @test Garamon._plan_cache_size(key,plan)>=Base.summarysize((key,plan))
    end

    # Numerically equal metrics can have different coefficient types. Reusing
    # a Float64 plan for a rational algebra would silently lose exactness.
    floating = algebra([1.0 0.0; 0.0 1.0])
    rational = algebra(Rational{Int}[1 0; 0 1])
    fa = multivector(floating, Dict(1 => 1.0); storage=:sparse)
    ra = multivector(rational, Dict(1 => 1//3); storage=:sparse)
    rb = multivector(rational, Dict(1 => 1//7); storage=:sparse)
    typed = ProductPlanCache()
    @test cached_product!(typed, fa, fa) == fa * fa
    exact = cached_product!(typed, ra, rb)
    @test eltype(exact) <: Rational
    @test coefficient_mask(exact, 0) == 1//21
    @test cache_stats(typed).misses == 2

    tiny = ProductPlanCache(max_bytes=1)
    @test cached_product!(tiny, a, b) == a * b
    @test cache_stats(tiny).entries == 0
    @test_throws ArgumentError ProductPlanCache(max_bytes=-1)
    sizing = ProductPlanCache(max_bytes=10_000)
    cached_plan!(sizing, a, b)
    size_ab = cache_stats(sizing).estimated_bytes
    empty!(sizing)
    cached_plan!(sizing, a, c)
    size_ac = cache_stats(sizing).estimated_bytes
    bounded = ProductPlanCache(max_bytes=max(size_ab, size_ac))
    cached_plan!(bounded, a, b)
    cached_plan!(bounded, a, c)
    @test cache_stats(bounded).entries == 1
    @test cache_stats(bounded).evictions == 1
    @test cached_product!(bounded, a, b) == a * b
    roulette = ProductPlanCache(max_bytes=max(size_ab, size_ac),
                                policy=:roulette, seed=42)
    for (left, right) in ((a, b), (a, c), (a, b), (a, a), (a, c))
        @test cached_product!(roulette, left, right) == left * right
        @test cache_stats(roulette).estimated_bytes <=
              cache_stats(roulette).max_bytes
    end
    @test cache_stats(roulette).evictions >= 1
    @test cache_stats(roulette).policy == :roulette
    tinylfu = ProductPlanCache(max_bytes=max(size_ab, size_ac),
                              policy=:tinylfu, frequency_slots=16)
    for _ in 1:4
        @test cached_product!(tinylfu, a, b) == a * b
    end
    @test cached_product!(tinylfu, a, c) == a * c
    @test cache_stats(tinylfu).admissions_rejected == 1
    @test cache_stats(tinylfu).entries == 1
    for _ in 1:5
        @test cached_product!(tinylfu, a, c) == a * c
    end
    @test cache_stats(tinylfu).evictions >= 1
    @test cache_stats(tinylfu).frequency_metadata_bytes == 64
    sieve = ProductPlanCache(max_bytes=max(size_ab, size_ac), policy=:sieve)
    @test cached_product!(sieve, a, b) == a * b
    @test cached_product!(sieve, a, b) == a * b
    @test cached_product!(sieve, a, c) == a * c
    @test cache_stats(sieve).sieve_scans >= 2
    @test cache_stats(sieve).entries == 1
    @test cache_stats(sieve).estimated_bytes <= cache_stats(sieve).max_bytes
    empty!(tinylfu)
    empty!(sieve)
    @test cache_stats(tinylfu).frequency_metadata_bytes == 64
    @test cache_stats(tinylfu).admissions_rejected == 0
    @test cache_stats(sieve).sieve_scans == 0
    @test_throws ArgumentError ProductPlanCache(policy=:tinylfu,
                                                frequency_slots=15)
    mixed_left = a + b
    mixed_right = a - b
    exact_mixed = mixed_left * mixed_right
    @test length(sparse(exact_mixed).values) > 1
    for _ in 1:5
        @test cached_product!(roulette, mixed_left, mixed_right) == exact_mixed
        @test cached_product!(roulette, b + c, a - c) ==
              (b + c) * (a - c)
    end
    empty!(roulette)
    @test cache_stats(roulette).entries == 0
    @test_throws ArgumentError ProductPlanCache(policy=:unknown)
    @test_throws ArgumentError ProductPlanCache(policy=:roulette, seed=-1)
    simultaneous = ProductPlanCache(max_bytes=10_000)
    tasks = [Threads.@spawn cached_product!(simultaneous, a, b) for _ in 1:16]
    @test all(fetch(task) == a * b for task in tasks)
    @test cache_stats(simultaneous).entries == 1
    empty!(cache)
    @test cache_stats(cache).entries == 0
    @test cache_stats(cache).hits == 0
end

@testset "Bounded full-grade wedge block" begin
    ga = algebra(Rational{Int}[2 0 0 0; 0 3 0 0; 0 0 4 0; 0 0 0 5])
    a = basisblade(ga, [1, 3]; storage=:sparse) +
        2basisblade(ga, [2, 4]; storage=:sparse)
    b = 3basisblade(ga, [2, 4]; storage=:sparse) -
        basisblade(ga, [1, 3]; storage=:sparse)
    plan = prepare_top_wedge(ga, 2)
    @test length(plan.left_masks) == 6
    @test top_wedge_coefficient(plan, a, b) ==
          coefficient(wedge(a, b), [1, 2, 3, 4])
    @test top_wedge_coefficient(plan, a, b) ==
          product_coefficient(a, b, [1, 2, 3, 4]; operation=:wedge)
    @test_throws ArgumentError prepare_top_wedge(ga, 2; max_slots=5)
    @test_throws ArgumentError prepare_top_wedge(ga, -1)
    @test_throws ArgumentError top_wedge_coefficient(plan, a +
                                                     basisvector(ga, 1; storage=:sparse), b)
    @test_throws ArgumentError top_wedge_coefficient(plan, a,
                                                     basisvector(ga, 1; storage=:sparse))
    basis(ga)[1] = "changed"
    @test_throws ArgumentError top_wedge_coefficient(plan, a, b)

    large = algebra(66, :ega)
    l = basisvector(large, 1; storage=:sparse)
    r = basisblade(large, collect(2:66); storage=:sparse)
    large_plan = prepare_top_wedge(large, 1; max_slots=66)
    @test length(large_plan.left_masks) == 66
    @test top_wedge_coefficient(large_plan, l, r) == 1
    word_boundary = algebra(64, :ega)
    boundary_plan = prepare_top_wedge(word_boundary, 1; max_slots=64)
    @test top_wedge_coefficient(boundary_plan,
          basisvector(word_boundary, 1; storage=:sparse),
          basisblade(word_boundary, collect(2:64); storage=:sparse)) == 1
    for n in 2:5, rgrade in 0:n
        small = algebra(n, :ega)
        grade_plan = prepare_top_wedge(small, rgrade)
        for left_mask in grade_plan.left_masks,
            right_mask in 0:((1 << n) - 1)
            count_ones(right_mask) == n - rgrade || continue
            x = multivector(small, Dict(left_mask => 1); storage=:sparse)
            y = multivector(small, Dict(UInt64(right_mask) => 1);
                            storage=:sparse)
            @test top_wedge_coefficient(grade_plan, x, y) ==
                  coefficient(wedge(x, y), collect(1:n))
        end
    end
end

@testset "Sorted sparse wedge coefficients" begin
    for n in 1:5
        ga = algebra(n, :ega)
        a = multivector(ga,
            Dict(UInt64(mask) => (mask % 3 + 1)//1
                 for mask in 0:((1 << n) - 1)); storage=:sparse)
        b = multivector(ga,
            Dict(UInt64(mask) => (2 - mask % 5)//1
                 for mask in 0:((1 << n) - 1)
                 if mask % 5 != 2); storage=:sparse)
        for target in 0:((1 << n) - 1)
            output = [i for i in 1:n if !iszero(target & (1 << (i - 1)))]
            @test sorted_wedge_coefficient(a, b, output) ==
                  product_coefficient(a, b, output; operation=:wedge)
        end
    end
    nonorth = algebra(Rational{Int}[2 1 0; 1 3 1; 0 1 0])
    x = basisvector(nonorth, 1; storage=:sparse) +
        basisvector(nonorth, 3; storage=:sparse)
    y = basisvector(nonorth, 2; storage=:sparse)
    @test sorted_wedge_coefficient(x, y, [1, 2]) ==
          coefficient(wedge(x, y), [1, 2])
    boundary = algebra(64, :ega)
    @test sorted_wedge_coefficient(
        basisvector(boundary, 64; storage=:sparse),
        basisvector(boundary, 1; storage=:sparse), [1, 64]) == -1
    @test_throws ArgumentError sorted_wedge_coefficient(x, y, [1, 1])
    @test_throws ArgumentError sorted_wedge_coefficient(x,
        basisvector(algebra(3, :ega), 2; storage=:sparse), [1, 2])
end

@testset "Bounded output-grade wedge block" begin
    for n in 2:5
        ga = algebra(n, :ega)
        for target in 0:((1 << n) - 1),
            rgrade in 0:count_ones(target)
            output = [i for i in 1:n if !iszero(target & (1 << (i - 1)))]
            plan = prepare_wedge_coefficient(ga, rgrade, output)
            @test length(plan.left_masks) == binomial(count_ones(target), rgrade)
            left = multivector(ga,
                Dict(UInt64(mask) => (-1)^mask * (mask + 1)
                     for mask in 0:((1 << n) - 1)
                     if count_ones(mask) == rgrade); storage=:sparse)
            right_grade = count_ones(target) - rgrade
            right = multivector(ga,
                Dict(UInt64(mask) => mask + 2
                     for mask in 0:((1 << n) - 1)
                     if count_ones(mask) == right_grade); storage=:sparse)
            @test wedge_coefficient(plan, left, right) ==
                  product_coefficient(left, right, output; operation=:wedge)
        end
    end
    ga = algebra(66, :ega)
    output = [1, 63, 64, 65, 66]
    plan = prepare_wedge_coefficient(ga, 2, output; max_slots=10)
    left = basisblade(ga, [1, 65]; storage=:sparse)
    right = basisblade(ga, [63, 64, 66]; storage=:sparse)
    @test wedge_coefficient(plan, left, right) ==
          product_coefficient(left, right, output; operation=:wedge)
    @test_throws ArgumentError prepare_wedge_coefficient(ga, 2, output;
                                                        max_slots=9)
    @test_throws ArgumentError prepare_wedge_coefficient(ga, 6, output)
    @test_throws ArgumentError prepare_wedge_coefficient(ga, 2, [1, 1])
    @test_throws ArgumentError wedge_coefficient(plan, left +
        basisvector(ga, 2; storage=:sparse), right)
    other = algebra([1 0; 0 1])
    @test_throws ArgumentError wedge_coefficient(plan,
        basisvector(other, 1), basisvector(other, 2))
end

@testset "Exact bounded Clifford tensor trains" begin
    for n in 1:5
        metric_matrix = zeros(Int, n, n)
        for i in 1:n
            metric_matrix[i, i] = i % 3 == 0 ? 0 : iseven(i) ? -2 : 3
        end
        ga = algebra(metric_matrix)
        a = separable_train(ga, [iseven(i) ? 2 : 1 for i in 1:n],
                           [i - 2 for i in 1:n])
        b = separable_train(ga, [i for i in 1:n],
                           [iseven(i) ? -1 : 2 for i in 1:n])
        c = separable_train(ga, ones(Int, n), [iseven(i) ? 0 : 1 for i in 1:n])
        ab = train_product(a, b)
        @test expand(ab) == expand(a) * expand(b)
        @test expand(train_product(ab, c)) == (expand(a) * expand(b)) * expand(c)
        for mask in 0:((1 << n) - 1)
            indices = [i for i in 1:n if !iszero(mask & (1 << (i - 1)))]
            @test coefficient(ab, indices) ==
                  coefficient(expand(a) * expand(b), indices)
        end
    end
    large = algebra(66, :ega)
    zero_a, one_a = ones(Int, 66), zeros(Int, 66)
    zero_b, one_b = ones(Int, 66), zeros(Int, 66)
    zero_a[1], one_a[1] = 0, 1
    zero_b[66], one_b[66] = 0, 1
    a = separable_train(large, zero_a, one_a)
    b = separable_train(large, zero_b, one_b)
    ab = train_product(a, b)
    @test coefficient(ab, [1, 66]) == 1
    @test coefficient(train_product(b, a), [1, 66]) == -1
    @test coefficient(ab, Int[]) == 0
    @test Base.summarysize(ab) < 1_000_000
    @test_throws ArgumentError train_product(a, b; max_rank=1)
    @test_throws ArgumentError train_product(a, b; max_entries=1)
    @test_throws ArgumentError expand(ab)
    nonorth = algebra([2 1; 1 3])
    x = separable_train(nonorth, [1, 1], [1, 1])
    @test_throws ArgumentError train_product(x, x)

    # Full-rank cores exercise the parity construction beyond rank-one inputs.
    mixed = algebra(Diagonal([1, -2, 0, 3, -1]))
    function rank_two_cores(seed)
        return NTuple{2,Matrix{Int}}[
            i == 1 ? ([seed seed + 1], [seed + 2 -seed]) :
            i == 5 ? (reshape([seed + 1, -seed], 2, 1),
                      reshape([2, seed - 1], 2, 1)) :
            ([1 seed; i 2], [seed -1; 2 i])
            for i in 1:5]
    end
    rank_two_a = CliffordTrain(mixed, rank_two_cores(2))
    rank_two_b = CliffordTrain(mixed, rank_two_cores(3))
    rank_two_product = train_product(rank_two_a, rank_two_b)
    full_rank_two = expand(rank_two_a) * expand(rank_two_b)
    @test expand(rank_two_product) == full_rank_two
    @test coefficient(rank_two_product, [1, 3, 5]) ==
          coefficient(full_rank_two, [1, 3, 5])
    @test maximum(size(core[1], 2) for core in rank_two_product.cores) == 8
    @test_throws ArgumentError train_product(rank_two_a, rank_two_b;
                                             max_rank=7)
end

@testset "Declarative descriptor and bounded expression capture" begin
    ga = @algebra begin
        scalar = Rational{Int}
        basis = (:east, :north)
        metric = [2 1; 1 3]
    end
    @test basis(ga) == ["east", "north"]
    @test !isdefined(@__MODULE__, :east)
    a, b = basisvectors(ga)
    @test a * b + b * a == 2 * scalar(ga, 1//1)
    @test evaluate(@ga a ∧ a) == zero(a)
    expression = @ga a * b + b ∧ a
    @test evaluate(expression) == a * b + wedge(b, a)
    @test evaluate(expression; output=Int[]) == scalarpart(evaluate(expression))
    @test evaluate(expression; output=[1, 2]) == coefficient(evaluate(expression), [1, 2])
    @test evaluate(expression; outputs=[Int[], [1, 2]]) ==
          [scalarpart(evaluate(expression)), coefficient(evaluate(expression), [1, 2])]
    @test_throws ArgumentError evaluate(expression; output=Int[], outputs=[Int[]])
    @test evaluate(@ga a * b * a) == a * b * a
    @test evaluate(@ga a + b + a) == a + b + a
    repeated = @ga a * b + a * b
    expression_plan = prepare_expression(repeated)
    @test length(expression_plan) == 4
    @test evaluate(expression_plan) == evaluate(repeated)
    @test evaluate(expression_plan; output=Int[]) ==
          evaluate(repeated; output=Int[])
    @test evaluate(expression_plan; outputs=[Int[], [1, 2], Int[]]) ==
          evaluate(repeated; outputs=[Int[], [1, 2], Int[]])
    @test_throws ArgumentError prepare_expression(repeated; max_nodes=3)
    @test_throws ArgumentError prepare_expression(repeated; max_nodes=0)
    nested = @ga (a * b + b * a) * (a * b + b * a)
    nested_plan = prepare_expression(nested)
    @test length(nested_plan) == 6
    @test evaluate(nested_plan) == evaluate(nested)
    @test evaluate(nested_plan; output=Int[]) == scalarpart(evaluate(nested))
    @test evaluate(nested_plan; outputs=[Int[], [1], [1, 2]]) ==
          [coefficient(evaluate(nested), indices)
           for indices in (Int[], [1], [1, 2])]
    @test_throws ArgumentError evaluate(expression_plan;
                                        output=Int[], outputs=[Int[]])
    original_a = a.values[2]
    a.values[2] = 2 * original_a
    @test evaluate(expression_plan) == evaluate(repeated)
    @test evaluate(expression_plan; output=Int[]) ==
          scalarpart(evaluate(repeated))
    a.values[2] = original_a
    diagonal = algebra(Rational{Int}[2 0 0; 0 -3 0; 0 0 0])
    x = multivector(diagonal,
        Dict(UInt64(0) => 1//1, UInt64(1) => 2//1,
             UInt64(3) => -1//1); storage=:sparse)
    y = multivector(diagonal,
        Dict(UInt64(0) => -2//1, UInt64(2) => 3//1,
             UInt64(4) => 1//1); storage=:sparse)
    z = multivector(diagonal,
        Dict(UInt64(1) => 1//1, UInt64(6) => -2//1);
        storage=:sparse)
    all_outputs = [[i for i in 1:3 if !iszero(mask & (1 << (i - 1)))]
                   for mask in 0:7]
    nested_product = prepare_expression(@ga (x * y + z) * (y * z - x))
    expected = evaluate(nested_product)
    @test evaluate(nested_product; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(expected, indices) for indices in all_outputs]
    @test evaluate(nested_product; output=[1, 3], strategy=:recursive) ==
          coefficient(expected, [1, 3])
    nested_wedge = prepare_expression(@ga (x ∧ y) ∧ (y + z))
    @test evaluate(nested_wedge; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(evaluate(nested_wedge), indices) for indices in all_outputs]
    nested_left = prepare_expression(@ga left_contraction(x * y, y + z))
    @test evaluate(nested_left; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(evaluate(nested_left), indices) for indices in all_outputs]
    nested_right = prepare_expression(@ga right_contraction(x + y, y * z))
    @test evaluate(nested_right; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(evaluate(nested_right), indices) for indices in all_outputs]
    nested_repeat = prepare_expression(@ga (x * y + x * y) * z)
    @test evaluate(nested_repeat; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(evaluate(nested_repeat), indices) for indices in all_outputs]
    nested_scalar = prepare_expression(@ga 2 * (x * y) - z)
    @test evaluate(nested_scalar; outputs=all_outputs, strategy=:recursive) ==
          [coefficient(evaluate(nested_scalar), indices) for indices in all_outputs]
    for plan in (nested_product, nested_wedge, nested_left, nested_right,
                 nested_repeat, nested_scalar)
        reference = evaluate(plan)
        @test evaluate(plan; outputs=all_outputs,
                       strategy=:recursive_grades) ==
              [coefficient(reference, indices) for indices in all_outputs]
    end
    @test evaluate(nested_product; output=[1, 3],
                   strategy=:recursive_grades) == coefficient(expected, [1, 3])
    # The middle product can have grades 0, 2 and 4, but only grade 2
    # contributes to the requested scalar after multiplication by grade 2.
    larger = algebra(Matrix{Rational{Int}}(I, 6, 6))
    grade_two = sum(basisblade(larger, [i, j]; storage=:sparse)
                    for i in 1:5 for j in i+1:6)
    sliced = prepare_expression(@ga (grade_two * grade_two) * grade_two)
    @test evaluate(sliced; output=Int[], strategy=:recursive_grades) ==
          scalarpart(evaluate(sliced))
    @test_throws ArgumentError evaluate(sliced; output=Int[],
                                        strategy=:recursive, max_support=15)
    @test evaluate(sliced; output=Int[], strategy=:recursive_grades,
                   max_support=15) == scalarpart(evaluate(sliced))
    @test evaluate(sliced; output=Int[], strategy=:join3) ==
          scalarpart(evaluate(sliced))
    for (u, v, w) in ((x, y, z), (z, x, y), (y, z, x)),
        indices in all_outputs
        @test triple_product_coefficient(u, v, w, indices) ==
              coefficient((u * v) * w, indices)
    end
    triple = prepare_expression(@ga (x * y) * z)
    @test evaluate(triple; outputs=all_outputs, strategy=:join3) ==
          [coefficient(evaluate(triple), indices) for indices in all_outputs]
    @test_throws ArgumentError triple_product_coefficient(x, y, z, Int[];
                                                           max_pairs=1)
    @test_throws ArgumentError evaluate(nested_product; output=Int[],
                                        strategy=:join3)
    @test_throws ArgumentError triple_product_coefficient(a, b, a, Int[])
    ga66 = algebra(66, :ega)
    u66 = basisvector(ga66, 1; storage=:sparse)
    v66 = basisvector(ga66, 66; storage=:sparse)
    large_triple = prepare_expression(@ga (u66 * v66) * u66)
    @test evaluate(large_triple; output=[66], strategy=:join3) ==
          coefficient(evaluate(large_triple), [66])
    for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 66, 70)
        exact_ga = algebra(Matrix{Rational{Int}}(I, n, n))
        p = scalar(exact_ga, 1//1; storage=:sparse) +
            basisblade(exact_ga, [1, n]; storage=:sparse)
        q = scalar(exact_ga, 2//1; storage=:sparse) +
            basisvector(exact_ga, n; storage=:sparse)
        r = scalar(exact_ga, 3//1; storage=:sparse) +
            basisvector(exact_ga, 1; storage=:sparse)
        exact_plan = prepare_expression(@ga (p * q) * r)
        exact_value = scalarpart((p * q) * r)
        @test triple_product_coefficient(p, q, r, Int[]) == exact_value
        for mode in (:recursive, :recursive_grades, :join3)
            @test evaluate(exact_plan; output=Int[], strategy=mode) ==
                  exact_value
        end
        s = scalar(exact_ga, 4//1; storage=:sparse) +
            basisblade(exact_ga, [1, n]; storage=:sparse)
        four_outputs = [Int[], [1], [n], [1, n], Int[]]
        @test Garamon.quadruple_product_coefficients(p, q, r, s,
                                                     four_outputs) ==
              [coefficient(((p * q) * r) * s, indices)
               for indices in four_outputs]
    end
    @test_throws ArgumentError Garamon.quadruple_product_coefficients(
        x, y, z, x, [Int[]]; max_pairs=1)
    @test_throws ArgumentError Garamon.quadruple_product_coefficients(
        x, y, z, x, [Int[]]; max_support=1)
    @test_throws ArgumentError Garamon.quadruple_product_coefficients(
        a, b, a, a, [Int[]])
    @test_throws ArgumentError evaluate(nested_product; output=Int[],
                                        strategy=:recursive, max_support=1)
    @test_throws ArgumentError evaluate(nested_product; output=Int[],
                                        strategy=:recursive, max_pairs=1)
    @test_throws ArgumentError evaluate(nested_product; output=Int[],
                                        strategy=:unknown)
    @test_throws ArgumentError evaluate(prepare_expression(@ga a * b);
                                        output=Int[], strategy=:recursive)
    other = basisvector(algebra(Rational{Int}[4 0; 0 5]), 1)
    mixed = @ga a + other
    @test_throws ArgumentError evaluate(mixed; output=[1])
    @test_throws ArgumentError macroexpand(@__MODULE__, :(@algebra begin unsupported = 1 end))
    @test_throws ArgumentError macroexpand(@__MODULE__, :(@ga sin(a)))
    syntax_error = try
        macroexpand(@__MODULE__, :(@ga sin(a)))
    catch error
        error
    end
    @test occursin("unsupported @ga operation: sin at", sprint(showerror, syntax_error))

    conformal = @algebra begin
        scalar = Rational{Int}
        basis = (:e₀, :e₁, :e₂, :e₃, :e∞)
        kind = :cga
        metric = [0 0 0 0 -1;
                  0 1 0 0 0;
                  0 0 1 0 0;
                  0 0 0 1 0;
                 -1 0 0 0 0]
    end
    e₀, e₁, e₂, e₃, e∞ = basisvectors(conformal)
    @test scalarpart(e₀ * e∞) == -1
    @test e₀ * e∞ == -scalar(conformal, 1//1) + wedge(e₀, e∞)
    @test wedge(basisblade(conformal, [1, 2, 3]),
                basisblade(conformal, [1, 2, 3, 4])) == zero(e₀)
end

@testset "Independent Clifford-word oracle" begin
    G = Rational{Int}[2 1 0; 1 3 1; 0 1 0]
    ga = algebra(G)

    function addscaled!(out, terms, factor)
        for (word, value) in terms
            next = get(out, word, 0//1) + factor * value
            if iszero(next)
                delete!(out, word)
            else
                out[word] = next
            end
        end
        return out
    end
    function reduce_word(word::Tuple)
        for pos in 1:(length(word) - 1)
            i, j = word[pos], word[pos + 1]
            i < j && continue
            removed = Tuple(word[k] for k in eachindex(word) if k != pos && k != pos + 1)
            if i == j
                result = Dict{Tuple,Rational{Int}}()
                return addscaled!(result, reduce_word(removed), G[i, i])
            end
            swapped = collect(word)
            swapped[pos], swapped[pos + 1] = j, i
            result = Dict{Tuple,Rational{Int}}()
            addscaled!(result, reduce_word(Tuple(swapped)), -1//1)
            addscaled!(result, reduce_word(removed), 2 * G[i, j])
            return result
        end
        return Dict{Tuple,Rational{Int}}(word => 1//1)
    end
    function antisym(indices::Vector{Int})
        output = Dict{Tuple,Rational{Int}}()
        function visit(prefix, remaining, sign)
            if isempty(remaining)
                addscaled!(output, reduce_word(Tuple(prefix)), sign // factorial(length(indices)))
                return
            end
            for pos in eachindex(remaining)
                next = [remaining[k] for k in eachindex(remaining) if k != pos]
                visit([prefix; remaining[pos]], next, isodd(pos - 1) ? -sign : sign)
            end
        end
        visit(Int[], indices, 1)
        return output
    end
    function oracle_product(amask, bmask)
        indices(mask) = Int[i for i in 1:3 if mask & (1 << (i - 1)) != 0]
        left, right = antisym(indices(amask)), antisym(indices(bmask))
        words = Dict{Tuple,Rational{Int}}()
        for (aword, avalue) in left, (bword, bvalue) in right
            addscaled!(words, reduce_word((aword..., bword...)), avalue * bvalue)
        end
        output = Dict{Int,Rational{Int}}()
        while !isempty(words)
            word = argmax(length, keys(words))
            coefficient = words[word]
            mask = sum(1 << (i - 1) for i in word; init=0)
            output[mask] = coefficient
            addscaled!(words, antisym(Int[word...]), -coefficient)
        end
        return output
    end

    for amask in 0:7, bmask in 0:7
        a = multivector(ga, Dict(amask => 1//1); storage=:sparse)
        b = multivector(ga, Dict(bmask => 1//1); storage=:sparse)
        expected = oracle_product(amask, bmask)
        for mask in 0:7
            @test coefficient_mask(a * b, mask) == get(expected, mask, 0//1)
        end
    end
end
