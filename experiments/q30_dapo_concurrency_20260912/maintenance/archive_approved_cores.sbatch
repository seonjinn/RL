#!/usr/bin/env bash
#SBATCH --job-name=sna.home-core-archive
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

readonly archive=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-dapo-concurrency-20260912/core-archive-20260914
readonly -a sources=(/home/sna/core.2285827 /home/sna/core.2285828)
mkdir -p "$archive"
chmod 700 "$archive"

for source_file in "${sources[@]}"; do
  case "$source_file" in
    /home/sna/core.2285827|/home/sna/core.2285828) ;;
    *) exit 2 ;;
  esac
  destination="$archive/${source_file##*/}"
  if [[ ! -e "$source_file" && -f "$destination" ]]; then
    printf 'Already archived: %s\n' "$destination"
    continue
  fi
  test -f "$source_file"
  test ! -L "$source_file"
  test "$(stat -c %u "$source_file")" = "$(id -u)"
  file -b "$source_file" | grep -q 'ELF .* core file'
  before="$(stat -c '%d:%i:%s:%Y:%Z' "$source_file")"
  printf 'Archiving %s (%s)\n' "$source_file" "$before"
  if [[ ! -e "$destination" ]]; then
    test ! -e "$destination.partial"
    cp --sparse=always --preserve=timestamps "$source_file" "$destination.partial"
    chmod 600 "$destination.partial"
    mv -n "$destination.partial" "$destination"
    test ! -e "$destination.partial"
  fi
  source_hash="$(sha256sum "$source_file" | cut -d ' ' -f 1)"
  destination_hash="$(sha256sum "$destination" | cut -d ' ' -f 1)"
  test "$source_hash" = "$destination_hash"
  test "$before" = "$(stat -c '%d:%i:%s:%Y:%Z' "$source_file")"
  printf '%s  %s\n' "$destination_hash" "$destination" >> "$archive/verified.sha256"
  printf 'VERIFIED source=%s destination=%s sha256=%s\n' "$source_file" "$destination" "$source_hash"
  rm -- "$source_file"
  printf 'REMOVED_HOME_COPY %s; recoverable from %s\n' "$source_file" "$destination"
done
printf 'ARCHIVE_COMPLETE\n'
