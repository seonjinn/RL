# OCI Locked Runtime Sync Retry Design

## Goal

Make OCI runtime staging tolerate short GitHub transport outages without
changing the locked dependency graph, source provenance, or published runtime
identity.

## Confirmed failure

Runtime staging runs one `uv sync --locked` inside a new content-addressed
stage. OCI job `6076502` failed while fetching the lock-pinned
Emerging-Optimizers commit `1effa026ff096b7fa1063ca2fba19d98be6e6cdf`.
The bounded retry on `cpu_datamover`, job `6077198`, fetched and built that
dependency but later failed while fetching the lock-pinned Transformer Engine
commit `04a76c84423d9a4eb2f2010ef6692e347326cc00`. Both errors were GitHub port
443 connection timeouts. Neither job reached a CUDA Graph test or allocated a
GPU.

## Chosen behavior

Keep the existing `uv sync --locked` command and run it at most three times.
All attempts execute inside one stage job and reuse the same job-local
`UV_CACHE_DIR`, so an object fetched by an earlier attempt remains available to
the next attempt. Wait five seconds after the first failure and ten seconds
after the second failure. Print the failed attempt number and the next delay.

Do not expose retry count, delay, dependency URL, or cache location as user
configuration. The limits are part of the staging implementation rather than
the content-addressed runtime identity. Container digest, `uv.lock` digest,
NeMo-RL, Bridge, MCore, and TE commits, Python and uv versions, feature set,
package exclusions, tests, CUDA architectures, and CPU count remain unchanged.

## Failure and publication contract

If any attempt succeeds, continue through dependency identity checks, root
tests, source-provenance revalidation, build-cache removal, read-only
conversion, and atomic marker publication.

If all three attempts fail, return the final `uv sync` status. The existing
EXIT trap must remove the incomplete stage root and partial marker. A failed
stage can never be used for GPU attestation.

The retry applies only to the locked environment sync. It does not retry test
failures, imports, source verification, read-only conversion, or marker
publication.

## Validation

Focused tests must prove:

- a fake locked sync that fails once and then succeeds is invoked exactly
  twice, preserves its cache between attempts, and proceeds;
- a fake locked sync that fails three times is invoked exactly three times and
  returns the last failure;
- the production command still contains `sync --locked` and the existing
  package exclusions;
- shell syntax, the complete experiment launcher suite, and whitespace checks
  pass.

After commit and push, create a new runtime attestation bound to the new outer
NeMo-RL commit. Run `sbatch --test-only` immediately before submission and
monitor the CPU-only stage for at least five minutes. Only after the stage is
`COMPLETED|0:0` may the four-GPU attestation and CUDA Graph diagnostics run.

## Explainer maintenance

Update the maintained CUDA Graph explainer from its versioned context and
regenerate the HTML. Classify jobs `6076502` and `6077198` as runtime-staging
network failures, not CUDA Graph correctness or performance evidence.
