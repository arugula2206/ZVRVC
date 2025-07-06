# server_modified.py (再生スキップ対応版)

import socket
import argparse
import numpy as np
import torch
import struct
import time

import FreeVC.convert_rt as convert_rt
from frcrn import initialize_frcrn, denoise

try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    VAD_ENABLED = True
    print("Silero VADモデルの読み込みに成功しました。")
except Exception as e:
    VAD_ENABLED = False
    print(f"警告: Silero VADモデルの読み込みに失敗しました。VAD機能は無効になります。エラー: {e}")

FRCRN_ENABLED = False 
HOST = '0.0.0.0'
PORT = 8080
MAX_SAMPLES_FOR_FRCRN = 16000 * 10 

def handle_client(conn, addr):
    print(f"\nクライアントが接続しました: {addr}")
    try:
        with conn:
            len_data = conn.recv(4)
            if not len_data:
                print(f"クライアント {addr} がデータ長を送らずに切断しました。")
                return
            
            input_len = struct.unpack('>I', len_data)[0]
            print(f"受信予定のデータサイズ: {input_len} バイト")

            input_data = b''
            while len(input_data) < input_len:
                packet = conn.recv(4096)
                if not packet:
                    break
                input_data += packet
            
            if len(input_data) != input_len:
                print(f"警告: 受信データサイズが一致しませんでした (期待値:{input_len}, 実際:{len(input_data)})")

            print(f"音声受信完了。変換処理を開始します...")

            is_speech = False
            if VAD_ENABLED:
                try:
                    input_wave_tensor = torch.from_numpy(
                        np.frombuffer(input_data, dtype=np.int16)
                    ).float() / 32768.0
                    
                    speech_timestamps = get_speech_timestamps(input_wave_tensor, vad_model, sampling_rate=16000)
                    
                    if speech_timestamps:
                        is_speech = True
                        print(f"VAD: 発話を検出しました。{speech_timestamps}")
                    else:
                        print("VAD: 発話を検出できませんでした。変換をスキップします。")
                except Exception as e:
                    print(f"VAD処理中にエラーが発生しました: {e}。VADをスキップして変換処理を続行します。")
                    is_speech = True
            else:
                is_speech = True
            
            # ▼▼▼ 変更点: is_speech の結果に応じて処理を分岐 ▼▼▼
            if is_speech:
                # 発話が検出された場合、通常通り変換
                if FRCRN_ENABLED:
                    input_wave_for_vad = np.frombuffer(input_data, dtype=np.int16)
                    input_wave_float = input_wave_for_vad.astype(np.float32) / 32768.0
                    denoised_wave_float = denoise(input_wave_float.copy())
                    denoised_chunk_bytes = (denoised_wave_float * 32767.0).astype(np.int16).tobytes()
                    processed_bytes = convert_rt.convert_voice(denoised_chunk_bytes)
                else:
                    processed_bytes = convert_rt.convert_voice(input_data)
                
                print(f"処理完了。クライアントに送信します... (サイズ: {len(processed_bytes)} バイト)")
                conn.sendall(struct.pack('>I', len(processed_bytes)))
                conn.sendall(processed_bytes)

            else:
                # 発話が検出されなかった場合は、データ長0を送信
                print("処理完了。クライアントに再生不要の信号（データ長0）を送信します。")
                conn.sendall(struct.pack('>I', 0))
            # ▲▲▲ 変更点 ここまで ▲▲▲

            print("送信完了。")

    except ConnectionResetError:
        print(f"クライアント {addr} との接続がリセットされました。")
    except Exception as e:
        print(f"クライアント {addr} との通信中にエラーが発生しました: {e}")
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")

# (warm_up_models, start_server, mainブロックは変更なしのため省略)
def warm_up_models(frcrn_enabled, n_warmup):
    if n_warmup <= 0: return
    print(f"AIモデルを{n_warmup}回ウォームアップしています...")
    try:
        dummy_chunk = np.zeros(16000 * 2, dtype=np.int16).tobytes()
        with torch.no_grad():
            for i in range(n_warmup): 
                print(f"  ウォームアップ実行中... ({i+1}/{n_warmup})")
                if frcrn_enabled:
                    input_wave_float = np.frombuffer(dummy_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    denoised_wave_float = denoise(input_wave_float)
                    _ = convert_rt.convert_voice((denoised_wave_float * 32767.0).astype(np.int16).tobytes())
                else:
                    _ = convert_rt.convert_voice(dummy_chunk)
        print("ウォームアップ完了。")
    except Exception as e:
        print(f"警告: ウォームアップ中にエラーが発生しました: {e}")

def start_server(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(1.0)
        
        print(f"サーバーが {HOST}:{PORT} で待機中です。(停止するには Ctrl+C を押してください)")
        
        try:
            while True:
                try:
                    conn, addr = s.accept()
                    handle_client(conn, addr)
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\n停止信号を受信しました。サーバーをシャットダウンします。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="発話単位の声質変換サーバー")
    parser.add_argument("--hpfile", type=str, default="FreeVC/configs/freevc-s.json", help="JSON設定ファイルへのパス")
    parser.add_argument("--ptfile", type=str, default="FreeVC/checkpoints/G_190000.pth", help="モデルチェックポイントファイルへのパス")
    parser.add_gument("--tgtwav", type=str, default="FreeVC/inputs/zundamon/recitation001.wav", help="目標話者のWAVファイルへのパス")
    
    parser.add_argument("--use-frcrn", action="store_true", help="FRCRNノイズ除去モデルを有効にします。")
    parser.add_argument("--warmup", type=int, default=10, help="ウォームアップの実行回数を指定します。(デフォルト: 10)")
    args = parser.parse_args()

    print("声質変換モデル(FreeVC)を初期化しています...")
    convert_rt.load_models(args.hpfile, args.ptfile, args.tgtwav)
    print("FreeVCの初期化が完了しました。")
    
    FRCRN_ENABLED = args.use-frcrn
    if FRCRN_ENABLED:
        print("ノイズ除去モデル(FRCRN)を初期化しています...")
        initialize_frcrn(torch.device("cuda"), nsamples=MAX_SAMPLES_FOR_FRCRN)
        print("FRCRNの初期化が完了しました。")
    else:
        print("FRCRNノイズ除去は無効です。")
    
    warm_up_models(FRCRN_ENABLED, args.warmup)
    
    start_server(args)
    
    print("サーバープログラムを終了します。")