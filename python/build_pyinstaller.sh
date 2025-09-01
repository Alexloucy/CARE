#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if [ ! -n "$VIRTUAL_ENV" ]; then
    echo "Not in python venv; activate with:"
    echo "  source .venv/bin/activate"
    exit 1
fi

models=(
    "models/vit_care.yml"
    "models/CARE_Traced.pt"
    "models/md_v1000.0.0-redwood.pt"
    "models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    "models/dino_species_classifier.pt"
    "models/dino_binary_classifier_v3.pt"
    "models/CARE_Traced_GPUv.pt"
)

for model in "${models[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/$model" ]; then
        echo "$model must exist. Contact project owners for model files."
    fi
done

# Check if dinov3 folder exists
if [ ! -d "${SCRIPT_DIR}/dinov3" ]; then
    echo "dinov3 folder must exist. Contact project owners for dinov3 files."
    exit 1
fi

# Don't use Conda; it's multiprocessing impelementation is broken.
conda info &> /dev/null && (echo "DO NOT REDISTRIBUTE CONDA PYTHON" ; exit 1)

pyinstaller \
    --noconfirm \
    --name care-detect-reid \
    --distpath ../care-electron/resources/ \
    --add-data models/vit_care.yml:models \
    --add-data models/CARE_Traced.pt:models \
    --add-data models/CARE_Traced_GPUv.pt:models \
    --add-data models/md_v1000.0.0-redwood.pt:models \
    --add-data models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth:models \
    --add-data models/dino_species_classifier.pt:models \
    --add-data models/dino_binary_classifier_v3.pt:models \
    --add-data dinov3:dinov3 \
    main.py
