# Remote Worktree Edit Scratch Area

This directory contains older scratch copies used while developing cluster patches.

Do not rsync or launch current SpecDec jobs from this tree. The current maintained
SpecDec-RL overlay is:

`experiments/eagle3_qwen3_235b/remote_patches/SpecDec-RL/`

In particular, `remote_worktree_edit/nemo_rl/models/generation/vllm/vllm_worker.py`
still contains an older batch-gate patch and does not include the current V7
first-draft scheduler gate fix.
