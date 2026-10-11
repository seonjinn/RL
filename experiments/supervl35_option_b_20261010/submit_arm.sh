#!/bin/bash
set -euo pipefail

arm=${1:?Usage: submit_arm.sh ARM}
case "$arm" in
    bf16-bf16|bf16-mxfp8|mxfp8-bf16params-off|mxfp8-bf16params-on|mxfp8-fp8params-off|mxfp8-fp8params-on) ;;
    *) echo "Unknown precision arm: $arm" >&2; exit 2 ;;
esac

launcher_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$launcher_dir" rev-parse --show-toplevel)
test -r "$launcher_dir/configs/$arm.yaml"
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "Commit all source and launcher changes before submitting" >&2
    exit 1
fi
source_commit=$(git -C "$repo" rev-parse HEAD)
upstream=$(git -C "$repo" rev-parse '@{upstream}')
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" "$upstream"; then
    echo "Push $source_commit before submitting" >&2
    exit 1
fi

brr=${MARS_BRR:-$HOME/.local/share/mars-cluster-access/bin/brr}
test -x "$brr"
workspace=${MARS_WORKSPACE:-supervl35-optionb-20261010}
launcher="launcher16-${source_commit:0:10}"
durable=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/supervl35-optionb-20261010
image=${NRL_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/images/vllm029-20261006/nemo_rl_main_aligned_20261007_7776525.sqsh}
account=${SLURM_ACCOUNT:-nemotron_sw_post}
run_tag=$(date -u +%Y%m%d%H%M%S)
name="sna-supervl35-${arm}-${run_tag}"

"$brr" push --dry-run oci-hsg "$launcher_dir" "$workspace/$launcher"
"$brr" push oci-hsg "$launcher_dir" "$workspace/$launcher"
BRR_JOB_TIMEOUT=120 "$brr" job submit oci-hsg \
    --workspace "$workspace" --image "$image" \
    --account "$account" --partition batch --qos normal \
    --nodes 16 --gpus-per-node 4 --ntasks-per-node 1 \
    --segment 8 --time-limit 04:00:00 \
    --job-name "$name" --container-name "$name" \
    --mounts /lustre:/lustre:ro --mounts "$durable:$durable:rw" \
    --mounts /home/svc-dss/cache:/home/svc-dss/cache:ro \
    --mounts /raid/scratch:/raid/scratch:rw \
    -- /bin/bash -lc "set -euo pipefail; export NRL_EXPERIMENT_CONFIG_DIR=$launcher/configs; bash $launcher/run_arms.sh $source_commit $arm"
