#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then
    echo "Usage: ./tune.sh <scene> <gamma>"
    echo "Example: ./tune.sh garbage 0.4"
    exit 1
fi

SCENE=$1
GAMMA=$2
DATA_ROOT="/Volumes/Extreme SSD/nerfbusters-dataset"

echo "Tuning $SCENE with gamma=$GAMMA..."

sam_env/bin/python 01_extract_objects.py \
  --scene_dir "$DATA_ROOT/$SCENE" \
  --ply_in "$DATA_ROOT/opensplat_results/${SCENE}_splat.ply" \
  --ply_out "$DATA_ROOT/test_extracted/${SCENE}_extracted.ply" \
  --gamma "$GAMMA"

echo "Done! You can now view test_extracted/${SCENE}_extracted.ply"
