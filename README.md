# NMT task

1. training & inference based on [nemo](https://github.com/NVIDIA/NeMo)

pytorch lightning framework

trained on wmt20-en-zh corpus

**using pretrained nemo NMT model**

```bash
bash utils/download_nemo.sh
```

2. convert pytorch model to onnx

    2.1 encoder

    2.2 decoder(infer version: non_cache + cache)

```bash
bash utils/nemo2onnx.sh
```

3. convert onnx model to tensorRT engine

dynamic inputs are supported

fp32 fp16 are supported, but trt has optimization bug for fp16, int8 is not supported yet

Dockder env is recommanded for runting tensorrt inference, due to the cuda execution error.

```bash
bash ./docker/build
```

```bash
bash ./docker/interactive.sh
```

run convertion

```bash
bash utils/onnx2trt.sh
```

4. tensorRT inference

before doing inference, please make sure tokenizer models are provided or you can simple using the nemo tokenizer by 

```bash
tar -xvf ./model_bin/nmt_en_zh_transformer6x6.nemo
```

please check the configs before runing

```bash
bash inference/trt_infer.sh
```

