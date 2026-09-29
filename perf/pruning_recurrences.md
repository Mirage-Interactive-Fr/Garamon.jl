# Speculative pruning of multivector recurrences

Run exclusively, after coordination with other performance campaigns:

```sh
timeout 900s julia --project=perf/controller perf/pruning_recurrences.jl OUTPUT.csv
```

This is a self-contained experimental coefficient kernel under `perf/`. It does
not change, dispatch to, or benchmark the exact public Garamon product kernel.
It asks whether skipping updates and correcting a recurrence can pay for its own
cost. Every ambient dimension is real in the independent basis-mask oracle;
the timed kernel uses a prepared local coefficient layout. Its active algebra
has up to four directions spread across the ambient basis, hence 4, 8 or 16
coefficients. Larger ambient dimension does not imply a dense 2^n workload.

## Grid

- Dimensions: 2,3,4,5,8,12,16,32,64,65,96,128.
- Diagonal signatures: positive, alternating negative active directions, and
  one null active direction. Structurally zero paths are removed for every method.
- Affine recurrence: `x_next = rho * a_t*x + scalar(1/65536)`, where
  `a_t=3/4+e_j/4` cycles through active directions for contracting/amplifying
  cases. In neutral cases `a_t=e_j`, or identity for a null direction, giving
  an exact signed permutation of coefficients. Horizons 8 and 256.
- Bilinear recurrence: `x_next = rho*x + x*x/32`. Horizons 1,4,8.
- `rho=1/2,1,9/4`, labelled contracting/neutral/amplifying. The affine linear
  part contracts distances by at most 1/2, preserves coefficient l2 distances,
  or amplifies them by at least 9/8, respectively. For the last bound,
  `||e_j*x||_2 <= ||x||_2` and the reverse triangle inequality give
  `||(3/4+e_j/4)*x||_2 >= ||x||_2/2` for every tested signature. The bilinear case
  has local derivative bound `rho + ||x||_1/16`; the labels name its linear part,
  not a global stability proof for the nonlinear recurrence.
- Five methods give 2700 measured cases. Another 540 long bilinear cases at
  horizon 256 are explicitly not admitted: repeated squaring can make the exact
  rational oracle exceed its 16384-bit integer budget.

## Algorithms and contracts

`exact` is Float64 with every mathematically nonzero path, not real arithmetic.
`threshold` discards contributions with absolute value below 1/16384.
`roulette` samples every contribution with independent probability $p$ in $(0,1]$
and multiplies retained contributions by $1/p$. The legacy experiment uses
$p=1/2$ by default; GaramonBench declares a separate probability grid. Its
one-step output is conditionally
unbiased. This does not prove an unbiased nonlinear final trajectory.

Both approximate methods still inspect and form every contribution. They test
omission of output updates, not a block bound that avoids entire multiplications.
All methods contain the same path counters and finite-state checks. Threshold
accumulates omitted contributions in a temporary defect buffer. That work is
included even when the buffer is subsequently discarded.

`deferred` transports an error state at each step and adds it to the approximate
state every 16 steps and at the end. The affine correction is `rho*a_t*error`;
the nonlinear correction includes `rho*error + (x*error+error*x+error*error)/32`.
It restores the mathematical rational result in qualification. Float64 summation
order differs, so it is not advertised as bitwise equal to the Float64 baseline.

`replay` runs thresholded segments, then recomputes the entire segment from its
last exactly replayed checkpoint. All replay work and final repayment are timed.
It must restore the same Float64 final result bit for bit. Intermediate approximate
errors remain observable and are included in the maximum error statistic.

## Oracles and timing

The independent Rational{BigInt} oracle enumerates ambient basis indices and
counts inversions directly. It never uses the prototype path table. Every exact
trajectory is checked against it. Deferred and replay rational results must equal
its final result exactly. Floating errors are evaluated against a 256-bit BigFloat
conversion of the rational reference; those numerical error values are diagnostic,
not outward-rounded certificates.

An instrumented qualification pass retains the complete pre-correction trajectory
and checks maximum error, as well as final error. Its restoration timer includes
transport, corrections and replay, but contains per-step timer overhead. This is
a diagnostic separate from the retained uninstrumented total time. The timed run
does not retain trajectory history; it returns the final coefficients and
counters while including buffer allocation, RNG
initialization, path traversal, checking, pruning and all required corrections.
Common fixture construction and exact-oracle validation are excluded; their costs
are recorded separately. Reflective `summarysize` accounting is performed only
in the diagnostic pass; the timed run sets `measure_buffers=false` and does
not report a buffer size. No first-call compilation cost is claimed.

BenchmarkTools uses 7 samples maximum, evals=1, seconds=0.03. Report actual sample
counts; p95 is descriptive. Cases have fixed order in a shared PerfChecker executor.
Roulette statistics use 32 independent seeds outside timing. Variance is the sum
of coordinate sample variances, bias is the infinity norm of empirical mean minus
truth, RMSE is Euclidean, and maximum error is over all sampled final coefficients.
These estimates do not certify rare-event probabilities. Timings use seed 1.

The correction-path counter counts the three bilinear product contributions in
`x*error + error*x + error*error`; it does not count the scalar `rho*error` updates.
Those scalar updates are nevertheless executed and included in total time.
Omission counters include numerically zero contributions. The common trajectory
buffer layout is also used by the baseline and is not a minimal-memory exact
kernel. This is a controlled algorithm experiment, not optimized-library throughput.

## Limits

Up to ten million candidate paths per reference trajectory, 16384 bits per oracle
integer, 64 MiB reachable trajectory buffers, 256 MiB Julia allocations per timed
trajectory, 2 GiB process peak RSS, and a 900-second campaign. Checks are cooperative;
the external timeout gives a hard wall-clock limit. The oracle's 10-second check
occurs after its independent trajectory has finished. Sources are fingerprinted
before and after the campaign. A temporary PerfChecker feature entrypoint is
required by its API and automatically removed.

A row marked pass validates the experimental protocol and its applicable exact
restoration assertions. It does not mean an approximate method is exact or meets
an unstated accuracy tolerance. Adoption requires a measured total-time benefit
at an explicit accuracy contract; methods without such benefit should be refused
for that tested workload.
