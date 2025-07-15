# ZVRVC
発話単位でリアルタイムな声質変換を実現する、クライアントサーバー型のアプリケーションです。AIモデルにはStarGANv2-vcを利用しています。

## 概要
クライアント (client_utterance.py) がマイクであなたの音声を録音し、サーバー (server.py) に送信します。サーバーは受け取った音声を変換エンジン (converter.py) を使って目標の話者の声に変換し、クライアントに返送します。クライアントは受け取った音声をスピーカーで再生します。

## 1. 環境構築
本システムはDockerコンテナ内でサーバーを動作させることを前提としています。

### 1.1. 必要なもの
Docker および NVIDIA Container Toolkit: GPUを利用してAIモデルを動作させるために必要です。事前にインストールしてください。

各種モデルファイル: 学習済みのStarGANv2-vc、HiFi-GAN、F0予測モデルのファイル群。

参照音声データ: 目標話者（例: ずんだもん）の音声ファイル。

### 1.2. ファイル配置
プロジェクトのルートディレクトリ（例: ZVRVC/）に、以下の構造でファイルとディレクトリを配置してください。特にモデルファイルの階層構造が重要です。

```bash
ZVRVC/
├── server.py               # サーバー本体
├── client_utterance.py       # クライアント
├── converter.py              # 変換エンジン
├── const.py                  # 話者リスト定義
├── config.json               # サーバー設定ファイル
├── requirements.txt          # 必要なPythonライブラリリスト
├── Dockerfile                # Dockerイメージ構築用ファイル
│
├── starganv2_vc/             # StarGANv2-vcのコードとモデル
│   ├── Models/
│   │   └── ita4jvs20_pre_alljp/
│   │       └── epoch_00294.pth
│   ├── Utils/
│   │   └── JDC/
│   │       └── ep50_200bat32lr5_alljp.pth
│   ├── Configs/
│   │   └── config.yml
│   └── Data/
│       └── ITA-corpus/
│           └── zundamon/
│               └── recitation127.wav # 参照音声
│
└── hifigan_fix/              # HiFi-GANのコードとモデル
    ├── checkpoints/
    │   └── g_07180000_2
    └── config_v1_mod_2.json
```

requirements.txt の内容例:
```
torch==2.1.0
torchaudio==2.1.0
numpy
librosa
soxr
pyyaml
munch
sounddevice
modelscope
# その他、frcrnに必要なライブラリ
```

Dockerfile の内容例:
```Docker
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "python3", "server.py", "--config", "config.json" ]
```

### 1.3. 設定ファイルの編集
config.json を開き、ご自身の環境に合わせて各モデルファイルへのパスが正しいかを確認してください。

config.json
```
{
  "stargan_model_dir": "ita4jvs20_pre_alljp",
  "stargan_model_name": "epoch_00294.pth",
  "f0_model": "ep50_200bat32lr5_alljp.pth",
  "f0_model_key": "model",
  "hifigan_config": "./hifigan_fix/config_v1_mod_2.json",
  "hifigan_model": "./hifigan_fix/checkpoints/g_07180000_2",
  "target_speaker_key": "zundamon127",
  "warmup": 50
}
```

### 1.4. Dockerイメージのビルド
プロジェクトのルートディレクトリで以下のコマンドを実行し、Dockerイメージをビルドします。

```bash
docker build -t zvrvc .
```

## 2. デモの実行方法
デモはサーバーとクライアントの2つを起動して行います。

### 2.1. サーバーの起動
まず、ターミナルで以下のコマンドを実行して、Dockerコンテナを起動します。これにより、バックグラウンドでサーバーが起動し、モデルの読み込みが始まります。

```bash
docker run -d --gpus all -p 8080:8080 --name zvrvc_server zvrvc
```

サーバーのログを確認し、モデルの読み込みが完了して待機状態になったことを確認します。

```bash
docker logs -f zvrvc_server
```

以下のようなメッセージが表示されれば、サーバーの準備は完了です。

```bash
>>>> サーバーが 0.0.0.0:8080 で待機中です。(停止するには Ctrl+C を押してください) <<<<
```

### 2.2. クライアントの起動
別のターミナルを開き、client_utterance.py を実行します。

ステップ A: オーディオデバイス名の確認（初回のみ）
まず、以下のコマンドで利用可能なオーディオデバイスの名前を確認します。

```bash
python client_utterance.py --list-devices
```

出力結果から、使用したいマイクとスピーカーの名前（またはその一部）を控えておきます。

ステップ B: スクリプトの編集
client_utterance.py をテキストエディタで開き、ファイルの先頭にある設定項目を編集します。

```python
# client_utterance.py

# --- ▼▼▼ 設定 ▼▼▼ ---
# 使用するデバイス名を部分的に指定してください (例: "Focusrite", "MacBook Pro Microphone")
# 空白のままにすると、OSのデフォルトデバイスが使用されます。
INPUT_DEVICE_NAME = "マイクの名前の一部"
OUTPUT_DEVICE_NAME = "スピーカーの名前の一部"

# ... (以下略) ...
```

ステップ C: クライアントの実行
編集したスクリプトを保存し、以下のコマンドで実行します。

```bash
python client_utterance.py
```

クライアントが起動し、「🎤 発話の開始を待っています...」と表示されたら、マイクに向かって話しかけてください。録音が自動で行われ、変換後の音声が指定したスピーカーから再生されます。