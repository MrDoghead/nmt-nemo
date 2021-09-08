# NMT task

1. training & inference based on [nemo](https://github.com/NVIDIA/NeMo)

  pytorch lightning framework

  Datasets

    News-Commentary http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-zh.tsv.gz

    WikiTitles - http://data.statmt.org/wikititles/v2/wikititles-v2.zh-en.tsv.gz

    WikiMatrix - http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-zh.langid.tsv.gz

    Backtranslated Chinese - http://data.statmt.org/wmt20/translation-task/back-translation/zh-en/news.translatedto.zh.gz
    
    Backtranslated English - http://data.statmt.org/wmt20/translation-task/back-translation/zh-en/news.en.gz

    CC-Aligned - http://www.statmt.org/cc-aligned/sentence-aligned/en_XX-zh_CN.tsv.xz

  Datasets2
    
    UNv1.0.en-zh

    WMT'18, WMT'19 and WMT'20 test sets

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

5. rough results

After applying inference on test dataset (1,000 texts), we measure the time for pytorch model and tensorrt model on some specific sessions. The time listed below is the average time per batch.

|model|preprocess time|encoder time|generator time|postprocess time|total latency|
|---:|---:|---:|---:|---:|---:|
|pytorch|0.0014197865752503275s|0.008084586990065873s|0.41727768997102976s|0.0016448273165151476s|435.9568956270814s|
|tensorRT|0.0009395092390477657s|0.0028762537082657217s|0.20571816717181354s|0.0017853414667770267s|211.64874472934753s|

For more info, check `transformer-tensorrt-inference.pdf`
