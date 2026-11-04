#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -J readout-ctrl
# Readout-weight control (section 2.4). At the layer that feeds the readout, the pruning score of every connection
# into a unit is multiplied by |that unit's readout weight to the target category|, so the late wiring could restate
# the similarity of the categories' readout weights. Readout geometry = cosine similarity of the categories' readout
# weight rows (signed, and of their absolute values) from the torchvision checkpoints the circuits were extracted from
# (all IMAGENET1K_V1). It runs analysis/stats/tree_contrast.py with the readout geometry in the
# place of the mean category representations: W|A = the wiring's tree share beyond the readout geometry, A|W = the
# reverse. Planted control: AlexNet fc7 with the readout rows shuffled across categories (the geometry's own share
# should fall to chance). summarize.py then writes the summary table.
#
# Restartable: a task whose output exists is kept, partial outputs of an interrupted attempt move into a fresh archive
# folder, each attempt appends to its own logs, and the script refuses a destination that overlaps any input.
#
# Environment:
#   READOUT_DEST    new or restartable output folder (required)
#   CC_STATS_OUT    statistics outputs: vj_inputs/, outgeom_inputs/<net>/matrices_logits.npz, xnet_<net>/pairs/
#                   (cluster data)
#   CC_AN_ROOT      the layer-capture analyses, which hold the VGG-11 and ResNet gram folders named in GD below
#                   (cluster data)
#   CC_REPO         this repository (required under Slurm, which runs a spooled copy of this script)
#   CC_PYTHON       python with numpy, pandas, scipy and torch (default: python3)
#   CC_CHECKPOINTS  torchvision checkpoints (default: ${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints)
#   CC_TAXONOMY     category tree (default: $CC_REPO/taxonomy/inputs/category_taxonomy.csv)
# Needs bash >= 4.3 (associative arrays, wait -n).
set -uo pipefail
die() { echo "readout control: $*" >&2; exit 4; }

if [ -z "${CC_REPO:-}" ]; then
  [ -z "${SLURM_JOB_ID:-}" ] || die "under Slurm, pass CC_REPO (the script runs from a spooled copy)"
  CC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
fi
D=${READOUT_DEST:?pass READOUT_DEST}; SO=${CC_STATS_OUT:?pass CC_STATS_OUT}; AN=${CC_AN_ROOT:?pass CC_AN_ROOT}
PY=${CC_PYTHON:-python3}; CK=${CC_CHECKPOINTS:-${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints}
TAX=${CC_TAXONOMY:-$CC_REPO/taxonomy/inputs/category_taxonomy.csv}
X=$CC_REPO/analysis/readout_control; ST=$CC_REPO/analysis/stats; OUT=$D/out; LG=$D/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# Each network's gram folder (GD) and pairs folder (PD), as they are named below CC_AN_ROOT and CC_STATS_OUT.
declare -A GD=([alexnet]=$SO/vj_inputs [vgg11]=$AN/out_v66_20261018T124435Z/s12_vgg11/grams
               [resnet18]=$AN/r18_s100_shadow_20261019T015315Z/s12_resnet18/grams
               [resnet50v1]=$AN/r50_v66_view_20261018T160823Z/s12_resnet50v1/grams)
declare -A PD=([alexnet]=$SO/vj_inputs [vgg11]=$SO/xnet_vgg11/pairs [resnet18]=$SO/xnet_resnet18/pairs
               [resnet50v1]=$SO/xnet_resnet50v1/pairs)
declare -A CKPT=([alexnet]=$CK/alexnet-owt-7be5be79.pth [vgg11]=$CK/vgg11-8a719046.pth
                 [resnet18]=$CK/resnet18-f37072fd.pth [resnet50v1]=$CK/resnet50-0676ba61.pth)
declare -A KEY=([alexnet]=classifier.6.weight [vgg11]=classifier.6.weight [resnet18]=fc.weight
                [resnet50v1]=fc.weight)

# The destination must overlap no input root, in either direction, before anything is written.
real() { "$PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
inside() { case "$1/" in "${2%/}"/*) return 0;; esac; return 1; }  # is $1 equal to or under $2?
DR=$(real "$D") || die "cannot resolve READOUT_DEST"
for root in "$SO" "$AN" "$CK" "$CC_REPO" "$(dirname "$TAX")" "${GD[@]}" "${PD[@]}"; do
  RR=$(real "$root") || die "cannot resolve input root $root"
  if inside "$DR" "$RR" || inside "$RR" "$DR"; then die "destination $DR overlaps input root $RR"; fi
done

mkdir -p "$OUT" "$LG" || die "cannot create $OUT and $LG"
# One log set per attempt, appended, never truncated.
ATTEMPT="${SLURM_JOB_ID:-local}-r${SLURM_RESTART_COUNT:-0}-$(date +%Y%m%dT%H%M%S)-$$"
echo "log: $LG/attempt-$ATTEMPT.log"; exec >> "$LG/attempt-$ATTEMPT.log" 2>&1
cd "$ST" || exit 4

# Partial outputs go to one archive folder per attempt, reserved with mktemp -d; a failed move stops the run.
ARCH=""
aside() {
  [ -e "$1" ] || return 0
  if [ -z "$ARCH" ]; then ARCH=$(mktemp -d "$D/partial-$ATTEMPT.XXXXXX") || die "cannot reserve an archive folder"; fi
  mv -- "$1" "$ARCH/" || die "cannot move partial output $1 into $ARCH"
  echo "moved partial $1 -> $ARCH/"
}
gen() {  # net inputs-name [shuffle-seed]; make_readout_geometry.py writes site_map_readout_abs.json last
  local g=$OUT/inputs_$2
  [ -e "$g/site_map_readout_abs.json" ] && return 0; aside "$g"
  "$PY" "$X/make_readout_geometry.py" --checkpoint "${CKPT[$1]}" --key "${KEY[$1]}" \
    --like "$SO/outgeom_inputs/$1/matrices_logits.npz" --gram-dir "${GD[$1]}" --out "$g" ${3:+--shuffle-seed $3}; }
tc() {  # net layer kind inputs-name (partial outputs were moved aside before the tasks started)
  local o=$OUT/tc_$4_$2_$3
  "$PY" "$ST/tree_contrast.py" --network "$1" --layer "$2" --gram-dir "${GD[$1]}" --pairs-dir "${PD[$1]}" \
    --trained-dir "$OUT/inputs_$4" --site-map "$OUT/inputs_$4/site_map_readout_$3.json" \
    --taxonomy "$TAX" --nperm-test 1000 --seed 0 \
    --out "$o/tree_contrast_$2.csv" >> "$LG/tc-$4-$2-$3.$ATTEMPT.out" 2>> "$LG/tc-$4-$2-$3.$ATTEMPT.err"
  echo "tc $4 $2 $3 exit $?"; }

echo "start $(date -Is) host $(hostname) attempt $ATTEMPT"
for net in alexnet vgg11 resnet18 resnet50v1; do gen $net $net || die "gen $net failed"; done
gen alexnet alexnet_shuf 0 || die "gen alexnet_shuf failed"
TASKS=("alexnet fc7 abs alexnet" "alexnet fc7 signed alexnet" "alexnet fc7 abs alexnet_shuf"
       "alexnet fc6 abs alexnet" "alexnet fc6 signed alexnet" "alexnet conv5 abs alexnet" "alexnet conv5 signed alexnet")
for L in fc7 fc6; do TASKS+=("vgg11 $L abs vgg11" "vgg11 $L signed vgg11"); done
for L in stage4.block1.conv2 stage4.block0.conv2; do TASKS+=("resnet18 $L abs resnet18" "resnet18 $L signed resnet18"); done
for L in stage4.block2.conv3 stage4.block1.conv3; do TASKS+=("resnet50v1 $L abs resnet50v1" "resnet50v1 $L signed resnet50v1"); done
TODO=()
for t in "${TASKS[@]}"; do
  set -- $t; o=$OUT/tc_$4_$2_$3
  if [ -e "$o/tree_contrast_$2.csv" ]; then echo "tc $4 $2 $3 kept"; else aside "$o"; TODO+=("$t"); fi
done
running=0
for t in ${TODO[@]+"${TODO[@]}"}; do
  tc $t &
  running=$((running + 1)); if [ $running -ge 8 ]; then wait -n; running=$((running - 1)); fi
done
wait
missing=0
for t in "${TASKS[@]}"; do set -- $t; [ -e "$OUT/tc_$4_$2_$3/tree_contrast_$2.csv" ] || { echo "missing tc_$4_$2_$3"; missing=$((missing + 1)); }; done
echo "end $(date -Is) tasks ${#TASKS[@]} missing $missing"; [ $missing -eq 0 ] || exit 3
