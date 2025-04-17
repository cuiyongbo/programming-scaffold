#!/usr/bin/env bash

cd `dirname $(readlink -f $0)`

python -m pip install transformers -i https://mirrors.aliyun.com/pypi/simple/
python3 onnx_exporter.py

#/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan --minShapes=input_ids:1x512,attention_mask:1x512 --optShapes=input_ids:16x512,attention_mask:16x512 --maxShapes=input_ids:32x512,attention_mask:32x512 --fp16 --verbose | tee conversion_bs16_dy.txt

# trtexec path in `nvcr.io/nvidia/tritonserver:xx-py3` container
# Note the arguments for dynamic shapes
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan  --fp16 --verbose --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:4x256,attention_mask:4x256 --maxShapes=input_ids:32x512,attention_mask:32x512
