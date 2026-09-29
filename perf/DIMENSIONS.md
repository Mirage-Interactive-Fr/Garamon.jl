# Contiguous dimension screening

Run the controller from the package root:

```sh
timeout 900s julia --project=perf/controller perf/explore_dimensions.jl /tmp/garamon-dimensions.csv
```

The default campaign admits every integer dimension 2–256 for `cap12`, and every
integer dimension in the ranges below for the other families. A second positional
argument reduces the upper dimension for a smoke run. `--isolated` uses the standard
PerfChecker worker executor instead of the shared-process screening executor.
`--dimensions=63,64,65,66,127,128,129,130` selects a boundary-focused subset
with the same fixture, oracle, and admission rules.
The output path is explicit; the campaign creates no results in the repository.
Temporary feature entrypoints are removed on normal exit.

| Family | Admitted dimensions | Terms per operand | Work represented |
|---|---|---|---|
| `cap12` | 2–256 | up to 13 | Scalar and up to 12 dispersed bivectors, touching high bits |
| `local12` | 2–70 | up to 13 | Scalar and first 12 bivectors in lexicographic order |
| `mixed12` | 2–70 | up to 13 | Dispersed vectors, bivectors and consecutive trivectors, plus scalar |
| `full_grade2` | 2–11 | up to 56 (cap 64) | Every bivector plus scalar |
| `dense` | 2–5 | up to 32 | Every blade |

Every admitted family/dimension has scalar and four-coefficient contracts, evaluated
with `full`, `recursive`, `recursive_grades`, and `join3`. The four masks are scalar,
`e1`, `e1 ∧ en`, and `en`. Odd requested coefficients are structural zeros for the
even families; this deliberately exercises grade pruning. The mixed family also
includes odd support. Each timed sample performs one batch, rather than eight
repeated identical scalar requests. These measurements therefore do not directly
compare to the earlier campaign's eight-request timings.

## Correctness

The oracle multiplies two dictionaries of Int64 coefficients by explicitly counting
inversions between the sorted basis-index lists, then multiplies the result by the
third dictionary. It calls no Garamon product, sign, indexing, or coefficient
function. All coefficients are small integers; the bound `27*s^3 < 2^53` proves that
each sum and contribution is exactly representable by the measured Float64 kernels.
The qualification requires exact equality, without tolerances, before and after
sampling in shared mode. The standard isolated executor uses its declared OracleSpec.
All scenarios use Euclidean diagonal metrics. This campaign does not qualify
non-diagonal, indefinite, or degenerate metrics.

## Resource envelope

Admission is checked before algebra and oracle construction:

- Logical metric entries: at most 65,536 for `cap12`, 4,900 for the other families.
  Dimension 257 would need 66,049 entries and is rejected for `cap12`.
  This is a chosen size envelope, not an intrinsic algorithmic ceiling or a claim
  of exhausted physical memory. EGA metrics are stored sparsely above dimension 8.
  At dimension 256 the fixture also enumerates 32,640 candidate bivectors before
  selecting 12; observed setup allocations account for that temporary storage.
- Oracle contribution pairs: conservative `s^2+s^3` bounds of 2,366 for capped
  families, 270,336 for the full bivector family, 33,792 for dense inputs.
  The CSV also records the actual `s^2 + nnz(AB)*s` count after exact cancellation.
- Recursive support: 2,197, 4,096 and 32 entries for capped, full-bivector and dense
  families, respectively. Pair budget: 262,144, or 1,048,576 for full bivectors.
- Shared screening checks observed fixture allocations against 256 MiB and observed
  per-batch allocations against 64 MiB (128 MiB for full bivectors). These are Julia
  allocated bytes, not retained memory or an OS memory limit.
- The shared executor checks a 900 s campaign deadline and 2 GiB process peak RSS
  between cases. These checks are cooperative: they cannot interrupt a running
  kernel. The external `timeout 900s` command supplies the hard wall-clock timeout.
  The isolated executor retains structural admission limits but does not use these
  shared-executor allocation/RSS checks; apply external process limits if needed.
- Each benchmark uses `evals=1`, at most 12 samples and a 0.05 s sampling target.
  BenchmarkTools' seconds parameter is not a preemptive timeout. Compilation,
  warmup, oracle checks and garbage collection can exceed it.

Excluded family/dimension combinations are written as `skipped` with an explicit
reason. They are not counted as correctness passes. Within the admitted envelope,
any oracle or allocation failure makes the suite fail.

## Interpretation and reproducibility

Shared mode uses PerfChecker's custom `executor` interface and returns qualified
`CheckerResult` records from BenchmarkTools trials. It shares Julia compilation and
process state; it is a screening campaign, not a standard isolated-worker result.
Fixtures contain immutable-by-convention operands, and the evaluation functions
allocate their own working state. Only the current fixture is retained by the
executor. Use isolated workers to confirm consequential crossover decisions.

The CSV includes setup wall time and setup allocations. Setup includes algebra,
operands, expression plan, and exact oracle; first calls also include compilation.
It is excluded from median/p95 evaluation timings and must be included for
one-shot application cost. CSV evaluation allocation columns come from
BenchmarkTools and do not include fixture construction.

The header records Julia version, executor mode, budgets, campaign wall time,
process peak RSS, and a SHA-256 over package Project.toml, all source files, and the
two campaign source files. Source changes during a run invalidate its measurements.
The package working tree may be dirty: the fingerprint identifies the actual
measured content rather than relying only on a Git commit.

Twelve samples are useful for screening, not enough to claim narrow confidence
intervals. Family order and strategy order are fixed. Run-order effects, thermal
state, shared-process compilation, and the structural-zero requests need to be
considered before turning a crossover into an automatic dispatch rule.

## First completed campaign, 2026-09-27

The shared-process campaign completed all 3,256 admitted cases with exact oracles
passing, plus 6,944 explicit admission skips. Wall time was 762.0 s; process peak
RSS was 891,408,384 bytes. The host was an Intel Core i7-12700, Julia 1.13.0,
one Julia thread. The external 900 s timeout and all declared observed resource
limits were respected. Samples per case were 9–12, with 12 in all capped families.

This campaign belongs to source fingerprint
`6c8d1553b869af56bc65240c73ceba5aa9a709dfaf496b1b26dabeabcaffa475`,
which was unchanged during measurement. Later reusable-workspace implementation
work is outside this baseline. Re-running against subsequently edited sources
will produce a different source fingerprint and must be treated as a new campaign.

Selected median batch times in microseconds:

| Family / dimension / outputs | Full | Recursive | Grade pruning | Join3 |
|---|---:|---:|---:|---:|
| cap12 / 2 / scalar | 1.880 | 9.784 | 14.625 | 5.175 |
| cap12 / 5 / scalar | 29.282 | 33.270 | 40.915 | 17.609 |
| cap12 / 64 / scalar | 211.540 | 34.440 | 41.438 | 20.530 |
| cap12 / 65 / scalar | 1,380.542 | 77.794 | 103.944 | 74.956 |
| cap12 / 256 / scalar | 1,643.726 | 130.339 | 130.336 | 87.574 |
| cap12 / 256 / four | 1,674.324 | 140.748 | 144.820 | 223.595 |
| full_grade2 / 11 / scalar | 4,915.856 | 277.834 | 339.755 | 269.655 |
| full_grade2 / 11 / four | 4,849.454 | 342.136 | 407.722 | 958.667 |

The scalar join benefits do not transfer automatically to a four-output batch.
At small dimensions, materializing the product can be faster. Near ties (for
example scalar cap12 at n=4) are not evidence for a dispatch threshold. The
UInt64-to-BigInt transition at n=65 is visible and merits a separate follow-up.

Maximum observed allocation bytes per timed batch were 3,029,344 (`cap12`),
1,950,080 (`local12`), 2,723,752 (`mixed12`), 18,538,232 (`full_grade2`) and
1,245,616 (`dense`). At n=256, the scalar fixture setup took 9.483 ms and allocated
18,110,976 bytes; full evaluation allocated 2,990,944 bytes, versus 109,944 for
join3. The largest setup observed was the first n=2 fixture, 552.748 ms and
55,092,504 bytes including compilation. These setup figures also include the
independent oracle and therefore are not expression-plan-only costs.
