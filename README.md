# Garamon.jl

An experimental geometric algebra kernel developed from the private Julia
prototype and a source audit of [Garamon C++](https://github.com/vincentnozick/garamon).
Its aim is to choose an algorithm from the requested output and actual support,
instead of expanding every multivector into all `2^n` coefficients.

## Current operations

- Symmetric metrics, including nonorthogonal and degenerate cases. Exact rational
  coefficients are supported by the reference product.
- Geometric and exterior products, left and right contractions, Hestenes inner,
  dot and scalar products, grade selection, reversion, grade involution,
  Clifford conjugation, metric-independent right
  complement, metric dual, and a checked inverse for small algebras.
- Dense storage for small dimensions and dictionary storage with `UInt64` or
  `BigInt` blade masks. Dense expansion requires an explicit coefficient budget.
- Requested product coefficients for diagonal metrics, without building the
  other outputs when the request set is small. A bounded full-grade block plan specializes the top
  coefficient of repeated homogeneous wedges.
- Reusable product plans for repeated diagonal-metric products with the same
  active supports, plus a homogeneous batch interface. A bounded Julia kernel
  can be generated on demand for at most 256 paths. An optional, byte-bounded
  cache stores structural plans and never stores input coefficients; LRU is
  the default and roulette eviction is experimental.
- C++ compatible zero-based homogeneous blade ranking and unranking without
  a global 2^n index table, plus grade-position coefficient access.
- Exterior linear maps applied on demand, and simple blades stored as a matrix
  of vectors. The factorized representation supports a requested coefficient,
  exterior product, linear map, and guarded expansion. Reflection chains act on
  vectors or factorized blades without expanding a versor.
- Explicit congruence diagonalization for real metrics, including indefinite
  and degenerate forms, with basis conversions and residual diagnostics.
- Coordinate subspace reduction when the operands use only selected basis
  directions, with checked projection and embedding.
- Exact binary tensor trains for diagonal metrics. A two-state parity
  construction multiplies trains with bounded rank and core storage, so a
  requested coefficient need not expand an exponentially large multivector.
- A bounded `@algebra` descriptor syntax and `@ga` expression capture. A
  requested coefficient or small set of coefficients of a sum of binary
  products can avoid full products. `prepare_expression` shares repeated pure
  subexpressions and supports targeted final products with compound operands.

| Path | Metric | Representation and limit | Output |
|---|---|---|---|
| Direct geometric/wedge/contractions | Any checked symmetric metric, including degenerate | Dense through 20D or sparse `UInt64` through 64D / `BigInt` above; product expansion budget | Full multivector |
| Targeted product coefficients | Direct XOR join for diagonal metric; general metric uses full reference product | Any multivector storage; request list can be batched | One or selected coefficients |
| Sorted wedge coefficient | Any symmetric metric | Sparse `UInt64` through 64D; one support sorted per call | One coefficient |
| Prepared product and cache | Diagonal metric | Same input supports and checked metric/basis; path and byte budgets | Full product or CPU batch |
| Full-grade product plan | Diagonal metric | Complete pair of homogeneous grades under slot/path budgets; operands may use subsets | Full geometric, wedge or contraction product |
| Grade-block wedge | Any symmetric metric | Homogeneous operands; `max_slots` bounds the requested grade block | One coefficient |
| Exact tensor train | Diagonal metric | Binary cores with `max_rank` and `max_entries`; no truncation | One coefficient, or guarded expansion |
| Congruence | Symmetric real metric, including indefinite and degenerate | Diagonalization through 16D by default; residual and pivot checks | Basis conversion |
| General inverse | Checked symmetric metric; singular elements rejected | Generic regular solve through 5D; larger proven reverse-based cases | Inverse multivector |

These paths use ordinary Julia scalar arithmetic. Exact results should use
`Rational{BigInt}` or `BigInt` when machine integer overflow is possible.
No GPU kernel is provided.

## Provenance and ownership

The private Julia package retains its MIT license. The audited C++ generator
is also [MIT licensed](https://github.com/vincentnozick/garamon/blob/6267161e35be57873fa553f266ef04f2ba32d2e5/LICENCE.txt)
and credits Stéphane Breuils and Vincent Nozick. The Julia product and tensor
train kernels implement the cited algebraic identities; no generated C++
source was copied into this package.

An algebra descriptor's metric and basis names are read during calculations;
callers must leave them unchanged after construction. `DenseMultiVector(ga,
values)` adopts the supplied vector, while `SparseMultiVector(ga, dict)` copies
its input dictionary and removes zeros. `dense`, `sparse`, and arithmetic
allocate new coefficient containers. Treat coefficient containers as stable
during an operation. A prepared plan checks its metric, basis and support when
run; a packed batch instead owns a coefficient snapshot taken when packed.

```julia
using Garamon

ga = algebra([2//1 1//1; 1//1 3//1])
e1, e2 = basisvector(ga, 1), basisvector(ga, 2)
e1 * e2 + e2 * e1 == 2 * scalar(ga, 1//1)

orthogonal = algebra(12, :ega)
a = basisvector(orthogonal, 1; storage=:sparse)
b = basisvector(orthogonal, 12; storage=:sparse)
product_coefficient(a, b, [1, 12]) == 1
sorted_wedge_coefficient(a, b, [1, 12]) == 1
product_coefficients(a, b, [Int[], [1, 12]]) == [0, 1]
plan = prepare_product(a, b)
run_product(plan, a, b) == a * b
generated = generate_product(plan; max_paths=64)
run_generated_product(generated, a, b) == a * b
grade_plan = prepare_grade_product(orthogonal, 1, 1)
run_grade_product(grade_plan, a, b) == a * b
packed = pack_product_batch(plan, [a, 2a], [b, 3b])
unpack_product_batch(packed, run_packed_batch(packed)) ==
    [a * b, 6(a * b)]
cache = ProductPlanCache(max_bytes=1 << 20)
cached_product!(cache, a, b) == a * b
roulette = ProductPlanCache(max_bytes=1 << 20, policy=:roulette, seed=42)
cached_product!(roulette, a, b) == a * b

top_plan = prepare_top_wedge(orthogonal, 1)
top_wedge_coefficient(top_plan, a,
    basisblade(orthogonal, collect(2:12); storage=:sparse)) == 1
output_plan = prepare_wedge_coefficient(orthogonal, 1, [1, 12])
wedge_coefficient(output_plan, a, b) == 1

many = algebra(66, :ega)
zero_bits, one_bits = ones(Int, 66), zeros(Int, 66)
one_bits[1:12] .= 1
ta = separable_train(many, zero_bits, one_bits)
tb = separable_train(many, zero_bits, one_bits)
tc = train_product(ta, tb; max_rank=256, max_entries=1 << 20)
coefficient(tc, [1, 3, 5, 7, 9, 11])

subspace = coordinate_subspace(a, b)
subspace_product(subspace, a, b) == a * b

decomposition = diagonalize_metric(ga)
oa, ob = to_orthogonal(decomposition, e1), to_orthogonal(decomposition, e2)
from_orthogonal(decomposition, oa * ob, ga) == e1 * e2

C = @algebra begin
    scalar = Rational{Int}
    basis = (:east, :north)
    metric = [2 1; 1 3]
end
east, north = basisvectors(C)
evaluate(@ga east * north + north * east; output=Int[]) == 2
evaluate(@ga east * north + north * east;
         outputs=[Int[], [1, 2]]) == [2, 0]
expr = @ga east * north + east * north + east * north
expr_plan = prepare_expression(expr; max_nodes=32)
evaluate(expr_plan; outputs=[Int[], [1, 2]]) == [3, 3]
```

`outermorphism(P, a, target)` treats the columns of `P` as the images of the
source basis vectors. It always preserves the exterior product. Set
`check_metric=true` when it must also preserve the geometric product. The
map can be rectangular or singular when metric preservation is not required.

## From the initial Julia prototype

`basis(ga)` still returns basis names. Use `basisvectors(ga)` for algebraic
values; `@algebra` deliberately does not define variables in the caller's
scope. Its fields use ordinary Julia assignments inside `begin ... end`.
The former placeholder `src/operators.jl` was removed: use the operations
exported by `Garamon` or the bounded `@ga` capture. No generated C++ code is
copied into this package; the kernel follows the algebraic identities audited
against the [Garamon C++ repository](https://github.com/vincentnozick/garamon).

## Bounds and status

`DenseMultiVector` has a hard dimension limit of 20, and `dense` defaults to a
budget of 65,536 coefficients. Public products default to a 65,536-term budget
for sparse results and nonorthogonal intermediates; `max_terms` changes it.
The general inverse uses a left-regular solve
only through dimension 5; larger special cases are accepted when the reversed
element is demonstrably an inverse. Product plans require diagonal metrics and
identical input supports when executed. A factorized blade is a simple blade;
arbitrary homogeneous multivectors need not be decomposable into one.
The `@ga` capture accepts `+`, `-`, `*`, `∧`, `wedge`, and the two contractions
on captured values; it does not parse arbitrary Julia code. Unsupported
capture syntax reports the source file and line. The macros do not bind basis
names in the caller's scope; use `basisvectors(ga)` explicitly. A metric dual
needs an invertible pseudoscalar, while a right complement remains defined for
degenerate metrics. Exact calculations with machine integers are subject to
their ordinary overflow; use `BigInt` or `Rational{BigInt}` when needed.
An `ExpressionPlan` reuses operator nodes only for the same captured leaf
objects and recalculates values on each evaluation. Coefficient arithmetic
must be pure, and leaves must not be mutated during an evaluation. Requested
coefficients of compound products materialize their operands by default.
`evaluate(plan; output=..., strategy=:recursive)` can instead propagate a
requested mask through nested sums, products, wedges and contractions in a
diagonal metric. This experimental mode caps intermediate support with
`max_support` and support pairing with `max_pairs`; it can be slower than
materialization, so it is never selected automatically.
`strategy=:recursive_grades` first propagates possible grades forward and
requested grades backward, then builds only the demanded support grades. It
can avoid a support-budget failure even when its analysis overhead makes it
slower. For a left-associated triple geometric product of multivector leaves,
`strategy=:join3` computes requested coefficients by joining two supports and
looking up the uniquely determined third mask. The same operation is exposed
as `triple_product_coefficient(a, b, c, output; max_pairs=...)`. It requires a
diagonal metric and never samples or omits an admissible triple. These modes
remain explicit until their end-to-end costs are measured for the workload.
Metric diagonalization defaults to a maximum dimension of 16 and reports the
condition number of its basis map for floating metrics. A nonzero pivot below
the default tolerance raises an error instead of silently changing the metric;
`rtol=0` requests an exact nonzero pivot decision for floating input. Coordinate subspace
plans reject inputs outside their selected support. Neither transformation is
chosen automatically: conversion and setup costs must be measured for the
intended workload.
The top-wedge plan enumerates one grade only and rejects a grade block larger
than `max_slots` (65,536 by default). It accepts homogeneous operands whose
grades sum to the algebra dimension. The plan cache is explicit; its LRU byte
budget is based on estimated Julia object size, and a cache hit has a lookup
cost. Callers that already hold a plan can use `run_product` directly.
Roulette changes only which plan is evicted. It never samples or omits
algebraic contributions, and it can lose badly when frequently used supports
change. Keep LRU unless a representative trace wins. The generated kernel
compiles on first use and can be slower than a prepared plan after compilation.
`prepare_grade_product` enumerates all blade pairs of two grades only after
checking `max_slots` and `max_paths`. `run_grade_product` accepts changing
subsets of those grades. Its plan can be large; an exact-support product plan
is generally smaller when the same sparse supports repeat.
`pack_product_batch` snapshots a homogeneous batch into contiguous coefficient
matrices. `run_packed_batch` returns a matrix of output coefficients ordered by
`plan.output_masks`; `unpack_product_batch` restores multivectors if needed.
Packing validates algebra and support and may dominate one-time batches of
distinct input objects. Reused packed data can be evaluated without repeated
dictionary lookup; no GPU execution is implied by this layout.
`ProductWorkspace(plan, a, b; max_bytes=...)` stores coefficient and output
buffers for repeated calls with the same supports. `run_product_values!` returns
its output vector in `plan.output_masks` order; `run_product!` returns a sparse
multivector owned by the workspace. Both outputs are overwritten on the next
call, so retain a snapshot with `copy(values)` or `sparse(result)` if needed.
Use one workspace per concurrent evaluation. Admission checks the retained
object size against the explicit byte budget and a snapshot of free RAM; it
does not cap JIT memory or later growth of arbitrary-precision coefficients.
For repeated small-dimensional sparse and dense workloads, PerfChecker measures
the preparation cost separately from the allocation-free warm calls.
`prepare_wedge_coefficient` applies the same bounded idea to any requested
output mask: it enumerates only `binomial(output_grade,left_grade)` possible
left blades, and both operands must be homogeneous. Contributions outside the
requested mask are ignored. Setup and block packing must be included in a
performance comparison.
`sorted_wedge_coefficient` is a separate single-output path for sparse
`UInt64` operands through dimension 64. It sorts one support each call, so it
can be preferable to a full grade block for one-off medium-dimensional
queries. It works for any symmetric metric because wedge is metric-free.
`CliffordTrain` uses one two-slice core per basis direction. `separable_train`
creates rank-one inputs; `train_product` is exact only for diagonal metrics,
and rejects ranks or core-entry counts beyond its budgets. It uses no rounding
or hidden truncation. A train can represent exponentially many nonzero blades;
`expand` therefore checks the full `2^n` slot count against `max_terms`
before materializing it. Tensor trains are an explicit structured path, not
the default storage for sparse inputs.
Sparse blade masks use `UInt64` through 64 dimensions, `UInt128` through 128,
and `BigInt` thereafter. The fixed-width middle tier avoids arbitrary-precision
mask objects for 65–128D; the performance boundary is measured separately.
`quadruple_product_coefficients` explicitly evaluates four-factor requested
outputs through two balanced pair products and a final exact XOR join, with
pair and support budgets. Its use remains a caller choice pending broader
cross-dimensional measurements.

No GPU kernel, weighted decision diagram, or general expression compiler is
claimed. Run `julia --project=. test/runtests.jl` for exact tests and
`julia --project=. benchmark/compare.jl` for local comparisons. The PerfChecker
suite in `perf/suite.jl` has correctness oracles for direct, prepared and
generated products in 4D, 12D and sparse 66D, four cache traces, and four
ways to evaluate a nested expression with one requested coefficient. It
includes plan construction and cache eviction in each measured workload;
Julia's first compilation cost must be measured separately. For a standalone
controller, instantiate `perf/controller` and call
`PerfChecker.run_suite_file("perf/suite.jl"; profile=:quick, reports="reports")`.
Benchmarks are input- and machine-specific.
The first cross-dimensional expression screening can be reproduced with
`julia --project=perf/controller perf/explore_dimensions.jl OUTPUT.csv`; it
uses temporary PerfChecker entrypoints and removes them after the campaign.
