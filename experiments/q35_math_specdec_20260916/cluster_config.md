---
cluster: oci-hsg
ssh_host: sna@sna-oci-hsg-cs.park.nvidia.com
user: sna
remote_cwd: /home/sna/nemorl-q35-math-specdec-20260916
partition: batch
account: coreai_dlalgo_nemorl
container_image: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
gpus_per_node: 4
---

Four nodes,16GB200 GPUs; all source in home, builds and staged inputs in
node-local scratch, durable artifacts in Lustre. Existing Qwen3 worktrees
and running jobs must remain unchanged.
