#!/bin/bash
set -euo pipefail
umask 077

recipe=$1
model=$2
dataset=$3
output=$4
mkdir -p "$output"
exec >"$output/inspection.log" 2>&1
date -u
hostname
id
for path in "$recipe" "$model/config.json" "$dataset"; do
    test -r "$path"
    stat -c '%n %s bytes %U:%G %a' "$path"
done
cp "$recipe" "$output/shared_recipe.yaml"
cp "$model/config.json" "$output/model_config.json"
sha256sum "$recipe" "$model/config.json" >"$output/input_sha256.txt"
head -c 1048576 "$dataset" >"$output/dataset_prefix.jsonl"
for name in tokenizer_config.json processor_config.json preprocessor_config.json chat_template.jinja model.safetensors.index.json; do
    if test -r "$model/$name"; then
        cp "$model/$name" "$output/$name"
    fi
done
printf '%s\n' INSPECTION_COMPLETE
