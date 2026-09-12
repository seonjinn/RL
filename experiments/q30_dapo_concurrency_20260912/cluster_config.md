---
cluster: oci-hsg
ssh_host: sna@sna-oci-hsg-cs.park.nvidia.com
user: sna
remote_cwd: /home/sna/nemorl-q30-dapo-concurrency-20260912
partition: batch
account: coreai_dlalgo_llm
container_image: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
gpus_per_node: 4
---

32 GPUs, eight 4-GPU nodes, segment size 4 (same as the parent DAPO cohort).
Source is isolated from existing queued or running worktrees.
