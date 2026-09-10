---
cluster: oci-hsg
ssh_host: sna@sna-oci-hsg-cs.park.nvidia.com
user: sna
remote_cwd: /home/sna/nemorl-bf16-flashinfer-specdec-latest-main-20260909
partition: batch
account: nemotron_n4_post
container_image: /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260909_7023221.sqsh
mounts:
  - /lustre:/lustre
  - /home:/home
  - /raid:/raid
gpus_per_node: 4
nodes: 4
subproject: q30-latest-main-bf16-flashinfer-specdec
---

The source branch starts at upstream NeMo-RL `main` commit
`fd45cb8e44782c6ad3e218f47415d42c976b793c`. Jobs record the exact experiment
commit and recursive submodule revisions in their durable result directory.
