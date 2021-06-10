#!/bin/bash
mkdir -p data
cd data

echo "Downloading processed Appareal and UTKFace datasets - the datasets ownership belongs to their respective authors. Consult README for license information and links."

if [ ! -f processed_data.zip ]; then
    wget https://files.seeedstudio.com//ml/processed_data.zip
fi

if [ ! -d processed_data ]; then
    unzip processed_data.zip
fi

