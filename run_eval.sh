#!/usr/bin/env bash

CHECKPOINTS=(
    "outputs_cloud/moco-v2/best.pt"
    "outputs_cloud/barlow/best.pt"
    "outputs_cloud/spark/best.pt"
)

FAILED=()

echo "========================================"
echo "  PadChest Probe"
echo "========================================"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "--- probe: $ckpt ---"
    if uv run python -m finetune.probe --checkpoint "$ckpt"; then
        echo "OK: probe $ckpt"
    else
        echo "FAILED: probe $ckpt"
        FAILED+=("probe:$ckpt")
    fi
done

echo ""
echo "========================================"
echo "  PadChest Finetune"
echo "========================================"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "--- finetune: $ckpt ---"
    if uv run python -m finetune.finetune --checkpoint "$ckpt"; then
        echo "OK: finetune $ckpt"
    else
        echo "FAILED: finetune $ckpt"
        FAILED+=("finetune:$ckpt")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All done — no failures."
else
    echo "Completed with ${#FAILED[@]} failure(s):"
    for f in "${FAILED[@]}"; do echo "  FAILED: $f"; done
    exit 1
fi
