#!/bin/bash

# Set the environment variable if needed
export OMP_NUM_THREADS=1

DIR=output/fully_quantized_training

mkdir -p ${DIR}

# Run the distributed training
python3 /workspace/I-ViT/quant_train.py \
--abits 4 \
--wbits 4 \
--gbits 4 \
--qdtype int8 \
--model deit_small \
--epochs 90 \
--weight-decay 1e-4 \
--batch-size 256 \
--data /data/ILSVRC2012 \
--seed 42 \
--distributed \
--world_size 4 \
--dist_url env:// \
--lr 1e-6 
# > ${DIR}/output.log 2>&1 &
