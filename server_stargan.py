# server.py

import socket
import argparse
import numpy as np
import torch
import struct
import json
import time

# 手順1で作成した変換エンジンをインポート
import converter

# VAD（発話検出）関連
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    VAD_ENABLED = True
    print("Silero VADモデルの読み込みに成功しました。")
    import torchaudio
except Exception as e:
    VAD_ENABLED = False
    print(f"警告: Silero VADモデルの読み込みに失敗しました。: {e}")

# --- グローバル変数 ---
HOST = '0.0.0.0'
PORT = 8080
config = None

def handle_client(conn, addr):
    """クライアントを処理する"""
    print(f"\nクライアントが接続しました: {addr}")
    try:
        with conn:
            len_data = conn.recv(4)
            if not len_data: return
            input_len = struct.unpack('>I', len_data)[0]
            input_data = b''
            while len(input_data) < input_len:
                packet = conn.recv(4096)
                if not packet: break
                input_data += packet
            
            print(f"音声受信完了。変換処理を開始します...")
            
            # (VAD処理は変更なし)
            is_speech = True
            # ... (必要に応じてVADのロジックをここに追加) ...

            if is_speech:
                processed_bytes = converter.convert_voice(input_data, config['target_speaker_key'])
                print(f"処理完了。クライアントに送信します... (サイズ: {len(processed_bytes)} バイト)")
                conn.sendall(struct.pack('>I', len(processed_bytes)))
                conn.sendall(processed_bytes)
            else:
                conn.sendall(struct.pack('>I', 0))
            print("送信完了。")
    except Exception as e:
        print(f"クライアント {addr} との通信中にエラーが発生しました: {e}")
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")

def start_server():
    print("モデルを初期化しています...")
    converter.initialize_models(config)
    
    # ウォームアップ
    if config.get('warmup', 0) > 0:
        print(f"AIモデルを{config['warmup']}回ウォームアップしています...")
        dummy_wav_bytes = (np.random.randn(48000) * 10000).astype(np.int16).tobytes()
        for i in range(config['warmup']):
            print(f"  ウォームアップ実行中... ({i+1}/{config['warmup']})")
            _ = converter.convert_voice(dummy_wav_bytes, config['target_speaker_key'])
        print("ウォームアップ完了。")

    # サーバー待機
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(1.0)
        print(f"\n>>>> サーバーが {HOST}:{PORT} で待機中です。(停止するには Ctrl+C を押してください) <<<<")
        try:
            while True:
                try:
                    conn, addr = s.accept()
                    handle_client(conn, addr)
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\n停止信号を受信しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="発話単位の声質変換サーバー")
    parser.add_argument('--config', type=str, default='config.json', help='設定ファイル(JSON)へのパス')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"★★エラー★★: 設定ファイル '{args.config}' が見つかりません。")
        exit()
    
    start_server()
    print("サーバープログラムを終了します。")