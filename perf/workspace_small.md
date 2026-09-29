# ProductWorkspace: small dimensions and actual reuse horizons

```sh
timeout 900s julia --project=perf/controller perf/workspace_small.jl OUTPUT.csv
```

Run while package sources are stable and other performance campaigns are idle.
The campaign uses PerfChecker's shared-process executor interface, exact oracles,
and BenchmarkTools. It records a source fingerprint before and after measurement;
changes invalidate the campaign. A `.partial` CSV is updated after each completed
case and renamed on successful campaign completion. The final CSV records failures
and admission skips as well as passes. An adjacent `-amortization.csv` records
model estimates and the first actually measured winning horizon.

## Workloads and contracts

Every integer dimension 2–12 is considered. `cap12` contains a scalar and up to
12 dispersed bivectors. `full_grade2` contains every bivector and a scalar. Both
use SparseMultiVector inputs. `dense` populates every coefficient in genuine
DenseMultiVector inputs; its admissible dimensions are limited by path count.
Four deterministic operand pairs share the same support but have different,
small nonzero integer coefficients. Batches cycle through these pairs, so workspace
refresh is exercised throughout actual loops of 1, 32, or 1024 calls.

Three methods have the same mathematical output contract: direct multiplication,
`run_product`, and `run_product!` each produce a complete multivector that is
consumed immediately. A coefficient sum is read on every iteration, and the batch
checksum is checked. No history of returned objects is retained. For dense inputs
the direct result is dense and planned results are sparse; both represent the same
complete multivector. A persistent-snapshot consumer would need an additional copy
for the reused workspace and is outside this contract.

`run_product_values!` is measured separately as a borrowed coefficient vector in
plan order, including structural zero slots. Its times and allocations are
reported, but its amortization columns deliberately say `not_comparable` against
the complete-multivector interface.

Each execution method has two measured phases:

- `steady`: the plan/workspace already exists; all validation and buffer refresh
  performed by the public evaluation API remain inside the timed region.
- `end_to_end`: build the required plan and workspace once, then execute the
  entire horizon. Common algebra, operand and oracle construction is excluded
  equally for every method and is reported separately as fixture setup.

Preparation-only cases measure the plan alone, workspace construction from an
existing plan, and plan plus workspace. A function barrier separates construction
from the horizon loop, so imprecise inference at construction cannot add dynamic
dispatch and boxing on every loop iteration.

## Resource limits

A plan may contain at most 65,536 paths. Each timed batch may execute at most
262,144 path contributions. This common comparison envelope is conservative for
the allocating direct kernel; it is not a claim that the workspace could not run
larger horizons. Resulting admitted grids are:

| Family | Horizon 1 | Horizon 32 | Horizon 1024 |
|---|---|---|---|
| cap12 | n=2–12 | n=2–12 | n=2–12 |
| full_grade2 | n=2–12 | n=2–12 | n=2–6 |
| dense | n=2–8 | n=2–6 | n=2–4 |

There are 600 execution cases and 87 preparation cases, with 192 explicit
execution skips. Workspace admission is 64 MiB of reachable objects according to
Base.summarysize, also subject to the API's free-memory rule. The CSV records
plan and workspace footprints; these include the plan and shared algebra reachable
from each object, and are not the incremental heap cost after accounting for all
sharing with caller-owned inputs.

Observed fixture allocations and batch allocations are each limited to 256 MiB.
The process checks a 2 GiB peak-RSS limit and 900 s deadline between cases. These
are cooperative checks, not operating-system enforcement. The external timeout
provides the hard wall-clock limit. Allocation bytes count Julia allocations,
not retained memory; process peak RSS is recorded separately.

BenchmarkTools uses evals=1, at most 12 samples and a 0.1 s target per case. The
seconds parameter cannot interrupt a single long sample. The actual sample count
and p95 are recorded, including cases with fewer than 12 samples.

## Exact oracle and amortization

The oracle uses Int64 dictionaries and explicit inversions between sorted basis
indices. It calls no Garamon product, sign, or index helper. Every complete result
for all four operand pairs is checked. The executed horizon checksum is verified
before and after sampling. Small coefficients and the batch-work bound imply
`9 * max_batch_paths < 2^53`, so all contributions and accumulated checksums are
exact in Float64. Equality has no numerical tolerance.

The amortization model divides the separately measured preparation median by the
measured per-call saving at each horizon. It is reported only when that saving is
positive, and remains a model: allocation pressure and GC can make batch cost
nonlinear. The observed horizon uses the actual end-to-end batch measurement;
"first winning horizon 32" means that 32 wins among tested points, not that 32 is
the exact mathematical crossover. A win at 1024 after a loss at 32 only locates
the crossover between those tested points, subject to measurement variability.

Cases have a fixed order. Twelve samples, and especially shorter trials, do not
support narrow confidence claims. Preparation includes public validation and
memory admission, but excludes first-call compilation after warmup. Fixture setup
figures can include compilation of first-seen types.

## Separate cold-start experiment

This warm campaign does not measure native-code persistence across processes.
A follow-up grid should separate isolated precompile construction, package load
in a fresh process with an existing cache, the first call on a known algebra,
first use of a new support/metric in a warm process, and repeated calls sharing a
plan and workspace. Record cache size and construction cost as well as latency.
Compare a finite representative precompile workload with the default package
cache and an optional application sysimage. Keep compiler options, Julia version,
CPU target, package revision and output ownership contract fixed. Run compilation
traces separately from retained timings. Do not delete the user's existing depot.
The companion design note `garamon-precompilation-conception.md` in the task's
outputs gives the proposed experiment grid and official Julia references.
