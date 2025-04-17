# how to profile bge-large-zh-v1.5 with tritonserver

- download models: [BAAI/bge-large-zh-v1.5](https://www.modelscope.cn/models/BAAI/bge-large-zh-v1.5/summary)
- convert model format to TensorRT: `bash generate_models.sh`
- prepare model repository
    - move `model.plan` to `bge-large-zh-tensor-rt/bge_large_zh_trt/1/`
    - move bge-large-zh-v1.5 tokenizer files to `bge-large-zh-tensor-rt/python_bge_large_zh_tokenizer/1/`
- start tritonserver: `tritonserver --model-store=/root/code/model-store --model-control-mode=explicit --load-model=bge_large_zh_ensemble`
- run pressure script: `../tritonserver_pressure_test.py`
- run notebook to analysis model performance
