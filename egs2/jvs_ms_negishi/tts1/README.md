# JVS MS (Multple Speakers) RECIPE

[JVSレシピ](../../jvs/tts1/README.md)を拡張して、複数話者の音声合成を行うレシピです。

## 藤江研環境での学習の方法

### 環境構築時の注意点

基本的な日本語音声合成の環境（pyopenjtalkなど）以外に，
 X-Vectorの抽出のためにKaldiが必要です．
`tools`フォルダで以下を実行します．

```
$ git clone --depth 1 https://github.com/kaldi-asr/kaldi
$ cd kaldi/tools
$ make -j 8
$ cd ../src
$ ./configure --shared
$ make depend -j 8
$ make -j 8
```

### Stage 5 （特徴抽出）まで実行

```
./run.sh \
    --stop-stage 5 --use_xvector true \
    \
    --g2p pyopenjtalk_prosody \
    --min_wav_duration 0.38 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/finetune_vits.yaml
```

### 事前学習モデルのダウンロード

```
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads espnet/kan-bayashi_jsut_vits_prosody
```

各パスを環境変数に設定（長いため）
```
# 事前学習モデルの設定ファイルのパス
$ export PRETRAINED_MODEL_CONFIG=`find downloads/ -name "config.yaml"`
# 事前学習モデルのパラメータのパス
$ export PRETRAINED_MODEL_FILE=`find downloads/ -name "*.pth"`
# 事前学習モデルのトークンリストのパス
$ export PRETRAINED_MODEL_TOKENS_TXT=`find downloads/ -name "tokens.txt"`
```

### トークンリストの修正

```
# 古いファイルのバックアップ
$ cp dump/22k/token_list/phn_jaconv_pyopenjtalk_prosody/tokens.txt{,.bak}
# 新しいファイル（事前学習モデルのトークリンスト）のコピー
$ cp $PRETRAINED_MODEL_TOKENS_TXT dump/22k/token_list/phn_jaconv_pyopenjtalk_prosody/tokens.txt
```

### ファインチューニング（Stage 6以降）の実行

```
./run.sh \
    --ngpu 1 \
    --stage 6 --use_xvector true \
    \
    --g2p pyopenjtalk_prosody \
    --min_wav_duration 0.38 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/finetune_vits.yaml \
    --train_args "--init_param ${PRETRAINED_MODEL_FILE}" \
    --tag finetune_xvector_vits_raw_phn_jaconv_pyopenjtalk_prosody \
    --inference_model train.total_count.ave_10best.pth
```

- 環境に応じて`--ngpu`オプションの数を変更してください．指定するのはGPUの枚数です．
- メモリエラーが生じて止まってしまう場合は，`./conf/tuning/finetune_vits.yaml`の`batch_bins`を小さくしてください．

### 学習結果を使った音声合成例

[test_gen.ipynb](./test_gen.ipynb)を参照してください．
