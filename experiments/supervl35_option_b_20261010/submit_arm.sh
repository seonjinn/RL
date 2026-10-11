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
launcher_commit=$(git -C "$repo" rev-parse HEAD)
source_commit=${NRL_EXPERIMENT_SOURCE_COMMIT:-c8734274ac5883cab9c942b88a37820805a6aeae}
git -C "$repo" cat-file -e "$source_commit^{commit}"
if ! git -C "$repo" merge-base --is-ancestor "$source_commit" "$launcher_commit" || \
   ! git -C "$repo" diff --quiet "$source_commit" "$launcher_commit" -- . ':(exclude)experiments/supervl35_option_b_20261010'; then
    echo "The prepared runtime does not match the experiment source; rebuild it for $launcher_commit" >&2
    exit 1
fi
upstream=$(git -C "$repo" rev-parse '@{upstream}')
if ! git -C "$repo" merge-base --is-ancestor "$launcher_commit" "$upstream"; then
    echo "Push $launcher_commit before submitting" >&2
    exit 1
fi

brr=${MARS_BRR:-$HOME/.local/share/mars-cluster-access/bin/brr}
test -x "$brr"
workspace=${MARS_WORKSPACE:-supervl35-optionb-20261010}
launcher="launcher32-${launcher_commit:0:10}"
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
    --nodes 32 --gpus-per-node 4 --ntasks-per-node 1 \
    --segment 8 --time-limit 04:00:00 \
    --job-name "$name" --container-name "$name" \
    --mounts /lustre:/lustre:ro --mounts "$durable:$durable:rw" \
    --mounts /home/svc-dss/cache:/home/svc-dss/cache:ro \
    --mounts /raid/scratch:/raid/scratch:rw \
    -- /bin/bash -lc "set -euo pipefail; export NRL_EXPERIMENT_CONFIG_DIR=$launcher/configs; bash $launcher/run_arms.sh $source_commit $arm"
