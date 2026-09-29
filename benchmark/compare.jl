using Garamon
using LinearAlgebra
using Random
using Statistics

# Run with: julia --project=. benchmark/compare.jl
# All calls are warmed before timing. Numbers describe this machine and these
# inputs only; preparation and execution are reported separately.
function measure(f; repetitions=200)
    for _ in 1:5
        f()
    end
    GC.gc()
    elapsed = Vector{Float64}(undef, repetitions)
    result = nothing
    for i in eachindex(elapsed)
        start = time_ns()
        result = f()
        elapsed[i] = (time_ns() - start) / 1_000
    end
    return round(median(elapsed); digits=2), Base.summarysize(result),
           Base.@allocated(f())
end

function report(label, f; repetitions=200)
    μs, result_bytes, allocated_bytes = measure(f; repetitions)
    println(rpad(label, 36), lpad(μs, 11), " μs  ",
            lpad(allocated_bytes, 11), " B allocated  ",
            lpad(result_bytes, 11), " B result")
end

# Benchmark-only bounded code generation from structural plan paths. The
# topology is a compile-time tuple; factors and all coefficients stay runtime.
@generated function unrolled_values(::Val{P}, factors::Vector{F},
                                    left::Vector{S}, right::Vector{S},
                                    output_count::Int) where {P,F,S}
    statements = [:(values[$oi] += factors[$i] * left[$ai] * right[$bi])
                  for (i, (ai, bi, oi)) in enumerate(P)]
    result_type = promote_type(F, S)
    return quote
        values = zeros($result_type, output_count)
        $(statements...)
        values
    end
end

function compile_bounded(plan::ProductPlan; max_paths::Int=64)
    length(plan.paths) <= max_paths ||
        throw(ArgumentError("unrolled program exceeds path budget"))
    topology = Val(Tuple((ai, bi, oi) for (ai, bi, oi, _) in plan.paths))
    factors = [factor for (_, _, _, factor) in plan.paths]
    return (plan=plan, topology=topology, factors=factors)
end

function generated_product(program, a::AbstractMultiVector,
                           b::AbstractMultiVector)
    plan = program.plan
    T = promote_type(eltype(a), eltype(b), eltype(program.factors))
    left = T[coefficient_mask(a, mask) for mask in plan.left_masks]
    right = T[coefficient_mask(b, mask) for mask in plan.right_masks]
    values = unrolled_values(program.topology, program.factors, left, right,
                             length(plan.output_masks))
    return multivector(a.algebra,
        Dict(mask => value for (mask, value) in zip(plan.output_masks, values)
             if !iszero(value)); storage=:sparse)
end

function scenario(n, count)
    ga = algebra(n, :ega)
    masks = UInt64[one(UInt64) << i for i in 0:(count - 1)]
    a = multivector(ga, Dict(m => Float64(i) for (i, m) in enumerate(masks));
                    storage=:sparse)
    b = multivector(ga, Dict(m => Float64(i + 1) for (i, m) in enumerate(masks));
                    storage=:sparse)
    plan = prepare_product(a, b)
    cache = ProductPlanCache(max_bytes=1 << 20)
    cached_product!(cache, a, b)
    println("n=$n, active support=$count per operand")
    report("full active-support product", () -> a * b)
    report("prepared product (execution)", () -> run_product(plan, a, b))
    report("cache hit + prepared execution", () -> cached_product!(cache, a, b))
    report("one requested coefficient", () -> product_coefficient(a, b, Int[]))
    for q in (n >= 12 ? (1, 8, 64) : (1, 8))
        requests = [[i for i in 1:min(n, 6) if (mask & (1 << (i - 1))) != 0]
                    for mask in 0:(q - 1)]
        targeted = product_coefficients(a, b, requests; strategy=:targeted)
        targeted == product_coefficients(a, b, requests; strategy=:full) ||
            error("requested coefficients disagree")
        report("$q outputs, targeted", () ->
               product_coefficients(a, b, requests; strategy=:targeted))
        report("$q outputs, full", () ->
               product_coefficients(a, b, requests; strategy=:full))
        report("$q outputs, automatic", () ->
               product_coefficients(a, b, requests; strategy=:auto))
    end
    expr = @ga a * b + b * a
    report("expression, scalar requested", () -> evaluate(expr; output=Int[]))
    expression_requests = [[i for i in 1:min(n, 6)
                            if (mask & (1 << (i - 1))) != 0]
                           for mask in 0:7]
    evaluate(expr; outputs=expression_requests) ==
        [coefficient(evaluate(expr), indices) for indices in expression_requests] ||
        error("expression multi-output request disagrees")
    report("expression, 8 outputs requested", () ->
           evaluate(expr; outputs=expression_requests))
    report("expression, full result", () -> evaluate(expr))
    repeated = @ga a * b + a * b + a * b + a * b
    expr_plan = prepare_expression(repeated)
    length(expr_plan) == 6 || error("expression DAG did not share repeated products")
    evaluate(expr_plan) == evaluate(repeated) ||
        error("prepared expression disagrees on full result")
    evaluate(expr_plan; outputs=expression_requests) ==
        evaluate(repeated; outputs=expression_requests) ||
        error("prepared expression disagrees on requested coefficients")
    report("repeated expression, ordinary", () -> evaluate(repeated))
    report("repeated expression, planned", () -> evaluate(expr_plan))
    report("repeated expression, 8 requests", () ->
           evaluate(repeated; outputs=expression_requests))
    report("planned expression, 8 requests", () ->
           evaluate(expr_plan; outputs=expression_requests))
    report("expression DAG preparation", () -> prepare_expression(repeated))
    nested = @ga (a * b + b) * (b * a + a)
    nested_plan = prepare_expression(nested)
    expected_nested = [coefficient(evaluate(nested_plan), indices)
                       for indices in expression_requests]
    evaluate(nested_plan; outputs=expression_requests,
             strategy=:recursive) == expected_nested ||
        error("recursive nested coefficients disagree")
    report("nested expression, materialized", () ->
           evaluate(nested_plan; outputs=expression_requests))
    report("nested expression, recursive", () ->
           evaluate(nested_plan; outputs=expression_requests,
                    strategy=:recursive))
    report("nested scalar, materialized", () ->
           evaluate(nested_plan; output=Int[]))
    report("nested scalar, recursive", () ->
           evaluate(nested_plan; output=Int[], strategy=:recursive))
    report("nested DAG preparation", () -> prepare_expression(nested))
    report("plan preparation", () -> prepare_product(a, b))
    program = compile_bounded(plan)
    first_generated_ms = @elapsed first_generated = generated_product(program, a, b)
    first_generated == a * b || error("bounded generated product disagrees")
    println("bounded generator first execution, including compilation: ",
            round(1_000 * first_generated_ms; digits=2), " ms")
    report("bounded generated execution", () -> generated_product(program, a, b))
    if n <= 12
        da, db = dense(a), dense(b)
        report("full dense product", () -> da * db)
    end
    println()
end

scenario(4, 4)
scenario(12, 8)
scenario(66, 8)

function homogeneous_grade_scenario(n, r; repetitions=20)
    ga = algebra(n, :ega)
    masks = UInt64[mask for mask in 0:((1 << n) - 1)
                   if count_ones(mask) == r]
    a = multivector(ga, Dict(mask => i % 5 + 1
                             for (i, mask) in enumerate(masks));
                    storage=:sparse)
    b = multivector(ga, Dict(mask => i % 7 + 1
                             for (i, mask) in enumerate(masks));
                    storage=:sparse)
    plan = prepare_grade_product(ga, r, r)
    run_grade_product(plan, a, b) == a * b ||
        error("complete homogeneous-grade plan disagrees")
    sparse_a = multivector(ga,
        Dict(mask => i % 5 + 1 for (i, mask) in enumerate(masks)
             if isodd(i)); storage=:sparse)
    sparse_b = multivector(ga,
        Dict(mask => i % 7 + 1 for (i, mask) in enumerate(masks)
             if iseven(i)); storage=:sparse)
    run_grade_product(plan, sparse_a, sparse_b) == sparse_a * sparse_b ||
        error("subset homogeneous-grade plan disagrees")
    println("n=$n, full grade-$r x grade-$r geometric product;")
    println("grade slots: ", length(masks), "/", length(masks),
            "; plan paths: ", length(plan.paths),
            "; plan bytes: ", Base.summarysize(plan))
    report("direct full-grade product", () -> a * b; repetitions)
    report("planned full-grade product", () -> run_grade_product(plan, a, b);
           repetitions)
    report("grade-plan preparation", () -> prepare_grade_product(ga, r, r);
           repetitions=min(repetitions, 10))
    report("direct half-support product", () -> sparse_a * sparse_b;
           repetitions)
    report("planned half-support product", () ->
           run_grade_product(plan, sparse_a, sparse_b); repetitions)
    exact_subset = prepare_product(sparse_a, sparse_b)
    run_product(exact_subset, sparse_a, sparse_b) == sparse_a * sparse_b ||
        error("exact-support plan disagrees")
    report("exact-subset plan preparation", () ->
           prepare_product(sparse_a, sparse_b);
           repetitions=min(repetitions, 10))
    report("exact-subset plan execution", () ->
           run_product(exact_subset, sparse_a, sparse_b); repetitions)
    if n == 12
        trace = [(
            multivector(ga,
                Dict(mask => i % 5 + 1 for (i, mask) in enumerate(masks)
                     if (i + 3t) % 8 < 4); storage=:sparse),
            multivector(ga,
                Dict(mask => i % 7 + 1 for (i, mask) in enumerate(masks)
                     if (i + 5t) % 8 < 4); storage=:sparse))
            for t in 1:8]
        all(run_grade_product(plan, x, y) == x * y for (x, y) in trace) ||
            error("changing-support grade trace disagrees")
        println("n=12, eight changing grade-3 support pairs")
        report("changing supports, direct", () ->
               [x * y for (x, y) in trace]; repetitions=5)
        report("changing supports, fresh plans", () ->
               [run_product(prepare_product(x, y), x, y)
                for (x, y) in trace]; repetitions=5)
        report("changing supports, grade plan", () ->
               [run_grade_product(plan, x, y) for (x, y) in trace];
               repetitions=5)
        report("changing supports, setup + run", () -> begin
            fresh = prepare_grade_product(ga, r, r)
            [run_grade_product(fresh, x, y) for (x, y) in trace]
        end; repetitions=5)
    end
end
homogeneous_grade_scenario(8, 2)
homogeneous_grade_scenario(12, 3; repetitions=10)

println("n=6, rank-one binary trains: same complete product")
train_ga = algebra(6, :ega)
train_a = separable_train(train_ga, ones(Int, 6), fill(2, 6))
train_b = separable_train(train_ga, fill(2, 6), ones(Int, 6))
train_amv, train_bmv = expand(train_a), expand(train_b)
expand(train_product(train_a, train_b)) == train_amv * train_bmv ||
    error("complete tensor-train product disagrees")
report("direct complete 6D product", () -> train_amv * train_bmv)
report("train construction + expansion", () ->
       expand(train_product(train_a, train_b)))
report("train product construction", () -> train_product(train_a, train_b))

println("n=66, rank-one trains with 12 active bits: one output")
train_ga = algebra(66, :ega)
zero_bits, one_bits = ones(Int, 66), zeros(Int, 66)
one_bits[1:12] .= 1
train_a = separable_train(train_ga, zero_bits, one_bits)
train_b = separable_train(train_ga, zero_bits, one_bits)
train_amv = multivector(train_ga,
                        Dict(BigInt(mask) => 1 for mask in 0:4095);
                        storage=:sparse)
train_output = [1, 3, 5, 7, 9, 11]
train_result = train_product(train_a, train_b)
coefficient(train_result, train_output) ==
    product_coefficient(train_amv, train_amv, train_output) ||
    error("high-dimensional tensor-train coefficient disagrees")
println("train bytes: ", Base.summarysize(train_result),
        "; explicit input bytes: ", Base.summarysize(train_amv))
report("direct sparse requested output", () ->
       product_coefficient(train_amv, train_amv, train_output))
report("train product construction 66D", () ->
       train_product(train_a, train_b))
report("train requested output 66D", () ->
       coefficient(train_result, train_output))

function train_rank_growth_scenario()
    ga = algebra(12, :ega)
    base = separable_train(ga, ones(Float64, 12), fill(0.5, 12))
    current = base
    println("n=12, repeated exact train products without rank compression")
    for power in 2:5
        previous = current
        current = train_product(previous, base; max_rank=32)
        println("power $power: maximum core rank ",
                maximum(size(core[1], 2) for core in current.cores),
                "; bytes ", Base.summarysize(current))
        report("train product power $power", () ->
               train_product(previous, base; max_rank=32))
    end
end
train_rank_growth_scenario()

Random.seed!(42)
ga = algebra(12, :ega)
factors = randn(12, 6)
blade = FactorizedBlade(ga, factors)
println("n=12, grade=6, generic simple blade")
println("factorized bytes: ", Base.summarysize(blade))
report("one factorized coefficient", () -> coefficient(blade, collect(1:6)))
report("expand all coefficients", () -> expand(blade))

# The trie implementation is shared with GaramonBench and loaded by Garamon.
println("n=12, 100 grade-6 blades per operand: materialized trie experiment")
grade_masks = UInt64[mask for mask in 0:((1 << 12) - 1) if count_ones(mask) == 6]
shuffle!(grade_masks)
ga = algebra(12, :ega)
left = multivector(ga, Dict(mask => randn() for mask in grade_masks[1:100]);
                   storage=:sparse)
right = multivector(ga, Dict(mask => randn() for mask in grade_masks[101:200]);
                    storage=:sparse)
lt, rt = build_trie(left, 12), build_trie(right, 12)
println("trie nodes: ", count_nodes(lt), " + ", count_nodes(rt),
        "; visited pairs/prefixes and useful leaves: ",
        wedge_visits(lt, rt, 0, 12), "; naive blade pairs: 10000")
expected = wedge(left, right)
actual = multivector(ga, trie_wedge(lt, rt, 12); storage=:sparse)
isapprox(coefficient(expected, collect(1:12)),
         coefficient(actual, collect(1:12)); rtol=1e-10) ||
    error("trie experiment disagrees with wedge reference")
report("direct active-support wedge", () -> wedge(left, right))
report("trie construction (both)", () -> (build_trie(left, 12), build_trie(right, 12)))
report("trie wedge execution", () -> multivector(ga, trie_wedge(lt, rt, 12);
                                                  storage=:sparse))
grade_plan = prepare_product(left, right; operation=:wedge)
isapprox(coefficient(run_product(grade_plan, left, right), collect(1:12)),
         coefficient(expected, collect(1:12)); rtol=1e-10) ||
    error("prepared grade wedge disagrees with reference")
report("explicit wedge plan preparation", () -> prepare_product(left, right;
                                                                  operation=:wedge))
report("explicit wedge plan execution", () -> run_product(grade_plan, left, right))
report("one top-grade coefficient", () ->
       product_coefficient(left, right, collect(1:12); operation=:wedge))

# A full grade-6 block has 924 slots. In increasing-mask (colex) order,
# complementing a grade-6 mask reverses this order, so its partner is at N+1-i.
grade_sign(mask) = begin
    complement = UInt64((1 << 12) - 1) ⊻ mask
    parity = false
    remaining = mask
    while !iszero(remaining)
        i = trailing_zeros(remaining)
        isodd(count_ones(complement & ((UInt64(1) << i) - 1))) &&
            (parity = !parity)
        remaining &= remaining - 1
    end
    parity ? -1.0 : 1.0
end
grade_order = sort(grade_masks)
signs = grade_sign.(grade_order)
function pack_grades(a::SparseMultiVector, b::SparseMultiVector,
                     order::Vector{UInt64})
    return (Float64[get(a.values, mask, 0.0) for mask in order],
            Float64[get(b.values, mask, 0.0) for mask in order])
end
left_block, right_block = pack_grades(left, right, grade_order)
function block_coefficient(signs::Vector{Float64}, left::Vector{Float64},
                           right::Vector{Float64})
    result = 0.0
    for i in eachindex(left)
        result += signs[i] * left[i] * right[length(right) + 1 - i]
    end
    return result
end
isapprox(block_coefficient(signs, left_block, right_block),
         coefficient(expected, collect(1:12));
         rtol=1e-10, atol=1e-10) ||
    error("grade-block top coefficient disagrees with wedge")
report("grade-block packing (both)", () -> pack_grades(left, right, grade_order))
report("grade-block top coefficient", () ->
       block_coefficient(signs, left_block, right_block))
top_plan = prepare_top_wedge(ga, 6)
isapprox(top_wedge_coefficient(top_plan, left, right),
         coefficient(expected, collect(1:12)); rtol=1e-10, atol=1e-10) ||
    error("public top-wedge plan disagrees with reference")
report("top-wedge plan setup", () -> prepare_top_wedge(ga, 6))
report("top-wedge plan execution", () ->
       top_wedge_coefficient(top_plan, left, right))

function sorted_grade_terms(mv::SparseMultiVector)
    entries = sort!(collect(pairs(mv.values)); by=first)
    return ([first(entry) for entry in entries],
            [last(entry) for entry in entries])
end
sorted_left_masks, sorted_left_values = sorted_grade_terms(left)
sorted_right_masks, sorted_right_values = sorted_grade_terms(right)
function sorted_top_coefficient(left_masks, left_values,
                                right_masks, right_values)
    result = 0.0
    for i in eachindex(left_masks)
        partner = UInt64((1 << 12) - 1) ⊻ left_masks[i]
        position = searchsortedfirst(right_masks, partner)
        if position <= length(right_masks) && right_masks[position] == partner
            result += grade_sign(left_masks[i]) *
                      left_values[i] * right_values[position]
        end
    end
    return result
end
isapprox(sorted_top_coefficient(sorted_left_masks, sorted_left_values,
                                sorted_right_masks, sorted_right_values),
         coefficient(expected, collect(1:12)); rtol=1e-10, atol=1e-10) ||
    error("sorted grade-list top coefficient disagrees")
report("sorted grade-list preparation", () ->
       (sorted_grade_terms(left), sorted_grade_terms(right)))
report("sorted grade-list coefficient", () ->
       sorted_top_coefficient(sorted_left_masks, sorted_left_values,
                              sorted_right_masks, sorted_right_values))
report("sorted grade-list full path", () -> begin
    lm, lv = sorted_grade_terms(left)
    rm, rv = sorted_grade_terms(right)
    sorted_top_coefficient(lm, lv, rm, rv)
end)
isapprox(sorted_wedge_coefficient(left, right, collect(1:12)),
         coefficient(expected, collect(1:12)); rtol=1e-10, atol=1e-10) ||
    error("public sorted wedge coefficient disagrees")
report("public sorted wedge coefficient", () ->
       sorted_wedge_coefficient(left, right, collect(1:12)))

function sorted_wedge_output(left_masks, left_values,
                             right_masks, right_values, target)
    result = 0.0
    for i in eachindex(left_masks)
        amask = left_masks[i]
        iszero(amask & ~target) || continue
        partner = target ⊻ amask
        position = searchsortedfirst(right_masks, partner)
        if position <= length(right_masks) && right_masks[position] == partner
            sign = 1.0
            remaining = amask
            while !iszero(remaining)
                direction = trailing_zeros(remaining)
                lower = (one(remaining) << direction) - one(remaining)
                isodd(count_ones(partner & lower)) && (sign = -sign)
                remaining &= remaining - one(remaining)
            end
            result += sign * left_values[i] * right_values[position]
        end
    end
    return result
end

function output_grade_scenario(n)
    ga = algebra(n, :ega)
    masks = UInt64[mask for mask in 0:255 if count_ones(mask) == 3]
    K = n <= 64 ? UInt64 : BigInt
    a = multivector(ga, Dict(K(mask) => randn() for mask in masks);
                    storage=:sparse)
    b = multivector(ga, Dict(K(mask) => randn() for mask in masks);
                    storage=:sparse)
    output = collect(1:6)
    block = prepare_wedge_coefficient(ga, 3, output)
    direct = product_coefficient(a, b, output; operation=:wedge)
    isapprox(wedge_coefficient(block, a, b), direct; rtol=1e-10, atol=1e-10) ||
        error("output-grade block disagrees with requested wedge coefficient")
    println("n=$n, full grade-3 block on 8 active directions, grade-6 output")
    report("one output, direct join", () ->
           product_coefficient(a, b, output; operation=:wedge))
    report("one output, grade-block setup", () ->
           prepare_wedge_coefficient(ga, 3, output))
    report("one output, grade-block execution", () ->
           wedge_coefficient(block, a, b))
    lm, lv = sorted_grade_terms(a)
    rm, rv = sorted_grade_terms(b)
    target = K((1 << 6) - 1)
    isapprox(sorted_wedge_output(lm, lv, rm, rv, target), direct;
             rtol=1e-10, atol=1e-10) ||
        error("sorted wedge output disagrees")
    report("one output, sorted-list setup", () ->
           (sorted_grade_terms(a), sorted_grade_terms(b)))
    report("one output, sorted-list execution", () ->
           sorted_wedge_output(lm, lv, rm, rv, target))
    report("one output, sorted-list full path", () -> begin
        lm, lv = sorted_grade_terms(a)
        rm, rv = sorted_grade_terms(b)
        sorted_wedge_output(lm, lv, rm, rv, target)
    end)
    if n <= 64
        isapprox(sorted_wedge_coefficient(a, b, output), direct;
                 rtol=1e-10, atol=1e-10) ||
            error("public sorted wedge output disagrees")
        report("one output, public sorted-list", () ->
               sorted_wedge_coefficient(a, b, output))
    end
end
foreach(output_grade_scenario, (12, 66))

println()
println("n=6, tridiagonal nonorthogonal metric: same complete product")
G = Matrix{Float64}(I, 6, 6)
for i in 1:5
    G[i, i + 1] = G[i + 1, i] = 0.125
end
nonorth = algebra(G)
na = multivector(nonorth, Dict((UInt64(1) << i) => Float64(i + 1)
                               for i in 0:5); storage=:sparse)
nb = multivector(nonorth, Dict((UInt64(1) << i) => Float64(7 - i)
                               for i in 0:5); storage=:sparse)
decomposition = diagonalize_metric(nonorth)
oa, ob = to_orthogonal(decomposition, na), to_orthogonal(decomposition, nb)
direct = na * nb
converted = from_orthogonal(decomposition, oa * ob, nonorth)
isapprox(dense(direct).values, dense(converted).values; rtol=1e-10,
         atol=1e-10) || error("orthogonal path disagrees with direct product")
report("direct nonorthogonal product", () -> na * nb)
report("congruence setup", () -> diagonalize_metric(nonorth))
report("convert both operands", () -> (to_orthogonal(decomposition, na),
                                      to_orthogonal(decomposition, nb)))
report("orthogonal product only", () -> oa * ob)
report("orthogonal product + return", () ->
       from_orthogonal(decomposition, oa * ob, nonorth))
report("whole converted product", () -> begin
    ca, cb = to_orthogonal(decomposition, na),
             to_orthogonal(decomposition, nb)
    from_orthogonal(decomposition, ca * cb, nonorth)
end)

println()
println("n=12, batch of 32 pairs reusing the same operand objects")
batchleft = [left for _ in 1:32]
batchright = [right for _ in 1:32]
report("direct wedge batch", () ->
       [wedge(batchleft[i], batchright[i]) for i in eachindex(batchleft)])
report("prepared wedge batch", () ->
       batch_product(grade_plan, batchleft, batchright))
packed_batch = pack_product_batch(grade_plan, batchleft, batchright)
packed_output = run_packed_batch(packed_batch)
unpack_product_batch(packed_batch, packed_output) ==
    batch_product(grade_plan, batchleft, batchright) ||
    error("packed batch disagrees with prepared multivector batch")
report("packed batch preparation", () ->
       pack_product_batch(grade_plan, batchleft, batchright))
report("packed batch coefficient kernel", () ->
       run_packed_batch(packed_batch))
report("packed batch + multivector return", () ->
       unpack_product_batch(packed_batch, run_packed_batch(packed_batch)))
report("packed whole preparation + return", () -> begin
    packed = pack_product_batch(grade_plan, batchleft, batchright)
    unpack_product_batch(packed, run_packed_batch(packed))
end)

println("n=12, batch of 32 distinct values with identical supports")
varying_left = [(1 + i / 32) * left for i in 1:32]
varying_right = [(2 - i / 64) * right for i in 1:32]
varying_packed = pack_product_batch(grade_plan, varying_left, varying_right)
unpack_product_batch(varying_packed, run_packed_batch(varying_packed)) ==
    batch_product(grade_plan, varying_left, varying_right) ||
    error("varying packed batch disagrees")
report("varying direct wedge batch", () ->
       [wedge(varying_left[i], varying_right[i]) for i in 1:32])
report("varying prepared wedge batch", () ->
       batch_product(grade_plan, varying_left, varying_right))
report("varying packed preparation", () ->
       pack_product_batch(grade_plan, varying_left, varying_right))
report("varying packed whole return", () -> begin
    packed = pack_product_batch(grade_plan, varying_left, varying_right)
    unpack_product_batch(packed, run_packed_batch(packed))
end)

println()
println("n=66, 8 active coordinate directions: same complete product")
ambient = algebra(66, :ega)
active = [1, 3, 9, 17, 32, 47, 64, 66]
active_masks = [big(1) << (i - 1) for i in active]
sa = multivector(ambient,
                 Dict(mask => Float64(j) for (j, mask) in enumerate(active_masks));
                 storage=:sparse)
sb = multivector(ambient,
                 Dict(mask => Float64(9 - j) for (j, mask) in enumerate(active_masks));
                 storage=:sparse)
subspace = coordinate_subspace(sa, sb)
subspace_product(subspace, sa, sb) == sa * sb ||
    error("coordinate subspace disagrees with ambient product")
la, lb = project_subspace(subspace, sa), project_subspace(subspace, sb)
report("ambient sparse product", () -> sa * sb)
report("coordinate subspace setup", () -> coordinate_subspace(sa, sb))
report("subspace conversions", () ->
       (project_subspace(subspace, sa), project_subspace(subspace, sb)))
report("local 8D product", () -> la * lb)
report("local product + lift", () -> lift_subspace(subspace, la * lb, ambient))
report("whole subspace product", () -> subspace_product(subspace, sa, sb))

println()
println("n=66, alternating two support pairs: 32 complete products")
other_active = [2, 4, 10, 18, 33, 48, 63, 65]
other_masks = [big(1) << (i - 1) for i in other_active]
sc = multivector(ambient,
                 Dict(mask => Float64(j) for (j, mask) in enumerate(other_masks));
                 storage=:sparse)
sd = multivector(ambient,
                 Dict(mask => Float64(9 - j) for (j, mask) in enumerate(other_masks));
                 storage=:sparse)
trace = [isodd(i) ? (sa, sb) : (sc, sd) for i in 1:32]
trace_cache = ProductPlanCache(max_bytes=1 << 20)
for (a, b) in trace
    cached_product!(trace_cache, a, b)
end
direct_trace(trace) = [a * b for (a, b) in trace]
fresh_trace(trace) = [run_product(prepare_product(a, b), a, b)
                      for (a, b) in trace]
cached_trace(cache, trace) = [cached_product!(cache, a, b)
                              for (a, b) in trace]
direct_trace(trace) == cached_trace(trace_cache, trace) ||
    error("cached trace disagrees with direct products")
report("trace direct", () -> direct_trace(trace))
report("trace fresh plans", () -> fresh_trace(trace))
report("trace LRU plan cache", () -> cached_trace(trace_cache, trace))
println("cache state: ", cache_stats(trace_cache))

# Boolean ZDD support-only baseline. The separate WeightedZDD below stores
# exact coefficients and performs a signed product by recursive DAG branches.
struct BooleanZDD
    nodes::Vector{NTuple{3,Int}}
    root::Int
end

function boolean_zdd(masks::Vector{UInt32}, n::Int)
    nodes = NTuple{3,Int}[]
    interned = Dict{NTuple{3,Int},Int}()
    function visit(items::Vector{UInt32}, bit::Int)
        isempty(items) && return 0
        bit > n && return 1
        zero_items, one_items = UInt32[], UInt32[]
        flag = UInt32(1) << (bit - 1)
        for mask in items
            push!(iszero(mask & flag) ? zero_items : one_items, mask)
        end
        lo = visit(zero_items, bit + 1)
        hi = visit(one_items, bit + 1)
        hi == 0 && return lo
        return get!(interned, (bit, lo, hi)) do
            push!(nodes, (bit, lo, hi))
            length(nodes) + 1
        end
    end
    return BooleanZDD(nodes, visit(masks, 1))
end

function zdd_contains(dag::BooleanZDD, mask::UInt32, n::Int)
    id = dag.root
    for bit in 1:n
        id == 0 && return false
        id == 1 && return iszero(mask >> (bit - 1))
        node_bit, lo, hi = dag.nodes[id - 1]
        present = !iszero(mask & (UInt32(1) << (bit - 1)))
        if node_bit == bit
            id = present ? hi : lo
        elseif present
            return false
        end
    end
    return id == 1
end

function zdd_cardinality(dag::BooleanZDD)
    memo = Dict(0 => 0, 1 => 1)
    function count(id)
        return get!(memo, id) do
            _, lo, hi = dag.nodes[id - 1]
            count(lo) + count(hi)
        end
    end
    return count(dag.root)
end

function zdd_scenario()
    n = 24
    regular = UInt32[(UInt32(1) << i) | (UInt32(1) << j) |
                     (UInt32(1) << k) | (UInt32(1) << l)
                     for i in 0:20 for j in (i + 1):21
                     for k in (j + 1):22 for l in (k + 1):23]
    random_support = Set{UInt32}()
    while length(random_support) < length(regular)
        push!(random_support, UInt32(rand(0:((1 << n) - 1))))
    end
    random_masks = collect(random_support)
    structured = boolean_zdd(regular, n)
    random_dag = boolean_zdd(random_masks, n)
    zdd_cardinality(structured) == length(regular) &&
        zdd_cardinality(random_dag) == length(random_masks) ||
        error("ZDD support cardinality disagrees")
    all(mask -> zdd_contains(structured, mask, n), regular) &&
        all(mask -> zdd_contains(random_dag, mask, n), random_masks) &&
        !zdd_contains(structured, UInt32(0), n) ||
        error("ZDD membership disagrees")
    println("n=24, equal-size regular grade-4 and random support families")
    println("support masks each: ", length(regular),
            "; ZDD nodes regular/random: ",
            length(structured.nodes), "/", length(random_dag.nodes),
            "; raw mask bytes: ", Base.summarysize(regular),
            "; DAG bytes: ", Base.summarysize(structured), "/",
            Base.summarysize(random_dag))
    report("regular ZDD construction", () -> boolean_zdd(regular, n);
           repetitions=10)
    report("random ZDD construction", () -> boolean_zdd(random_masks, n);
           repetitions=10)
    report("regular ZDD membership", () ->
           zdd_contains(structured, regular[end], n))
    report("random ZDD membership", () ->
           zdd_contains(random_dag, random_masks[end], n))
end
zdd_scenario()

function weighted_zdd_scenario()
    Q=Rational{BigInt}
    n=6
    diagonal=Q[1,-1,2,0,1,-2]
    masks=[(big(1)<<(i-1)) | (big(1)<<(j-1)) for i in 1:n for j in i+1:n]
    left_terms=Dict(mask=>Q(isodd(i) ? 1 : -2) for (i,mask) in enumerate(masks))
    right_terms=Dict(mask=>Q(mod(i,3)+1) for (i,mask) in enumerate(masks))
    left=weighted_zdd(n,left_terms)
    right=weighted_zdd(n,right_terms)
    product=weighted_zdd_product(left,right,diagonal)
    ga=algebra(Diagonal(diagonal))
    direct=geometric_product(multivector(ga,left_terms;storage=:sparse),
        multivector(ga,right_terms;storage=:sparse))
    weighted_zdd_terms(product)==Dict(BigInt(mask)=>Q(value) for (mask,value) in direct.values) ||
        error("weighted ZDD coefficient product disagrees with direct product")
    println("weighted ZDD n=$n nodes left/right/product: ",
        length(left.nodes),"/",length(right.nodes),"/",length(product.nodes))
    report("weighted ZDD exact product DAG",()->weighted_zdd_product(left,right,diagonal);
        repetitions=10)
    report("weighted ZDD full extraction",()->weighted_zdd_terms(product);
        repetitions=10)
end
weighted_zdd_scenario()

# Census of the actual generated-kernel type key, not a count of numerical
# products or native-code bytes. Distinct absolute supports may map to one
# path topology because masks and metric factors are runtime data.
function generated_topology_census()
    println("dimension,blades_per_operand,plans,distinct_generated_topologies")
    for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 66, 70)
        ga = algebra(n, :ega)
        K = n <= 64 ? UInt64 : BigInt
        masks = [(one(K) << (i - 1)) | (one(K) << (j - 1))
                 for i in 1:n-1 for j in i+1:n]
        count = min(4, length(masks))
        rng = MersenneTwister(6000 + n)
        topologies = Set{Any}()
        for _ in 1:100
            left_indices = randperm(rng, length(masks))[1:count]
            right_indices = randperm(rng, length(masks))[1:count]
            left = multivector(ga,
                Dict(masks[i] => 1.0 for i in left_indices); storage=:sparse)
            right = multivector(ga,
                Dict(masks[i] => 1.0 for i in right_indices); storage=:sparse)
            plan = prepare_product(left, right)
            generated = generate_product(plan; max_paths=16)
            push!(topologies, typeof(generated).parameters[1])
        end
        println(n, ",", count, ",100,", length(topologies))
    end
end
generated_topology_census()
