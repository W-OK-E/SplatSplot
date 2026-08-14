
#!/usr/bin/env bash
set -euo pipefail

SCENES=("aloe" "art" "car" "century" "flowers" "garbage" "picnic" "pikachu" "pipe" "plant" "roses" "table")
DATA_ROOT="${1:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Starting SAM2 Batch Mask Generation..."
echo "Log file: sam2_batch.log"

for scene in "${SCENES[@]}"; do
    echo "======================================"
    echo "Starting $scene at $(date)"
    echo "======================================"
    
    # Check if masks directory already exists and has ALL files
    IMG_COUNT=$(ls -1q "$DATA_ROOT/$scene/images" 2>/dev/null | wc -l || echo 0)
    if [ "$IMG_COUNT" -eq 0 ]; then
        IMG_COUNT=$(ls -1q "$DATA_ROOT/$scene/images_4" 2>/dev/null | wc -l || echo 0)
    fi
    
    MASK_COUNT=$(ls -1q "$DATA_ROOT/$scene/masks" 2>/dev/null | wc -l || echo 0)
    
    if [ "$MASK_COUNT" -gt 0 ] && [ "$MASK_COUNT" -eq "$IMG_COUNT" ]; then
        echo "All $MASK_COUNT masks for $scene already exist. Skipping..."
        continue
    elif [ "$MASK_COUNT" -gt 0 ]; then
        echo "Partial masks found for $scene ($MASK_COUNT / $IMG_COUNT). Regenerating..."
    fi
    
    # Run the Python script for the scene
    # Spawning a new Python process for each scene ensures PyTorch memory is completely freed 
    # between scenes, preventing Out-Of-Memory (OOM) errors during the 10-hour run.
    "$PYTHON_BIN" generate_masks_sam2.py "$scene"
done

echo "======================================"
echo "ALL SCENES COMPLETED at $(date)!"
echo "======================================"
