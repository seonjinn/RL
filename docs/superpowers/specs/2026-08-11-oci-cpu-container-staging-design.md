# OCI CPU Container Staging Design

## Goal

Stage the exact NeMo-RL nightly image already attested on ptyche into OCI-HSG
without allocating idle GPUs. Preserve the current immutable image, metadata,
SHA256, and smoke-test workflow.

## Current constraint

OCI-HSG rejects GPU-free jobs in `batch` and rejects one-GPU jobs with
`QOSMinGRES`; a batch staging job would therefore reserve four GPUs only to
download and convert an image. OCI provides `cpu` and `cpu_datamover`
partitions for this non-GPU work.

## Chosen approach

Extend `stage_enroot_image.sbatch` so `PARTITION` accepts only `batch`, `cpu`,
or `cpu_datamover`, while retaining `batch` as the default. The wrapper passes
the selected partition to `sbatch` and continues to omit GPU, exclusive-node,
and memory requests. CPU partitions request 32 CPUs and four hours so large
nightly squashfs imports do not serialize on one CPU or hit the prior two-hour
limit. The default `batch` path does not add a CPU request. Production
experiment launchers remain restricted to `batch` and are not changed.

The OCI staging run will use `cpu_datamover`. Its source is the same immutable
nightly digest used on ptyche:

- image: `nvcr.io/nvidian/nemo-rl:nightly`;
- digest: `sha256:09509475e2efdef6f6bc32726f16b2cfbf238e7128246dbf27cb17d4472c401d`;
- image source commit: `0e687e6d07623d780a4174310e92382ce738a8a2`;
- expected squashfs SHA256:
  `67f63772db4e11bdae16d646706aec0ec49a5fd2f7c400ee62875ab869cf49b1`.

## Validation and error handling

Focused launcher tests will prove that:

- the default remains a GPU-free `batch` submission;
- `cpu_datamover` renders a GPU-free submission;
- CPU staging renders 32 CPUs and a four-hour limit;
- an unapproved partition fails before contacting Slurm; and
- digest, commit, and credential-free image validation remain unchanged.

Submission will follow `git pull --ff-only`, FairShare inspection, and
`sbatch --test-only`. The staging job will be monitored for at least five
minutes. Publication remains atomic: an import or integrity failure must not
replace the stable symlink or publish incomplete metadata.

After staging completes, the OCI artifact's digest, metadata, source commit,
and SHA256 must match the ptyche artifact. Only then will the existing GPU
runtime attestation run, followed by the five-step `moe_router` diagnostic
with three successful optimizer warmups and checkpointing disabled.

## Alternatives not selected

- Reserving four OCI GPUs for image conversion risks idle-resource alerts and
  provides no experimental value.
- PBSS transfer preserves the same bytes but requires an additional upload
  from ptyche, whose interactive authentication is currently unavailable.
- Reusing the older OCI nightly would confound the cross-cluster diagnostic
  with a software-stack difference.
