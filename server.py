# server.py

import socket
import argparse
import numpy as np
import time 

# FreeVCディレクトリ内のconvert_rtモジュールを、「convert_rt」という名前でインポートする
import FreeVC.convert_rt as convert_rt

# --- ネットワーク設定 ---
HOST = '0.0.0.0'  # 利用可能な全てのネットワークインターフェースで待機
PORT = 8080       # 待機するポート番号
CHUNK = 4096      # クライアントから受信するデータサイズ（クライアント側と合わせる必要あり）

# --- クロスフェード設定 ---
OVERLAP_SAMPLES = 256 

# --- 無音検出設定 ---
VAD_THRESHOLD = 150 # 無音と判断する音声エネルギー(RMS)のしきい値。この値より小さいと無音と見なす。(調整可能)

def handle_client_connection(conn, addr):
    """
    一人のクライアントとの接続と通信を専門に処理する関数。
    """
    print(f"\nクライアント接続処理を開始: {addr}")
    try:
        with conn:
            conn.settimeout(1.0)
            
            data_buffer = b''
            previous_processed_wave = np.zeros(CHUNK // 2, dtype=np.float32)
            
            hanning_window = np.hanning(OVERLAP_SAMPLES * 2).astype(np.float32)
            fade_out = hanning_window[OVERLAP_SAMPLES:]
            fade_in = hanning_window[:OVERLAP_SAMPLES]
            
            idle_timeout_counter = 0
            MAX_IDLE_TIMEOUTS = 5 # 5秒間データが来なければ接続を終了する

            while True:
                try:
                    data = conn.recv(CHUNK) 
                    if not data:
                        print("クライアントが接続を正常に閉じました。ループを抜けます。")
                        break 
                    
                    idle_timeout_counter = 0
                    
                    data_buffer += data
                    
                    while len(data_buffer) >= CHUNK:
                        process_chunk = data_buffer[:CHUNK]
                        data_buffer = data_buffer[CHUNK:]
                        
                        # --- 無音検出(VAD)処理 ---
                        input_wave_for_vad = np.frombuffer(process_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(np.square(input_wave_for_vad.astype(np.float64))))

                        if rms < VAD_THRESHOLD:
                            # 無音と判断した場合、AIモデルをバイパスして無音データをそのまま返す
                            processed_bytes = process_chunk 
                        else:
                            # --- FreeVC声質変換処理 ---
                            # ノイズ除去を行わず、直接変換する
                            processed_bytes = convert_rt.convert_voice(process_chunk)
                        
                        current_wave = np.frombuffer(processed_bytes, dtype=np.int16).astype(np.float32)

                        tail = previous_processed_wave[-OVERLAP_SAMPLES:]
                        head = current_wave[:OVERLAP_SAMPLES]
                        blended_part = (tail * fade_out) + (head * fade_in)
                        
                        output_wave = np.concatenate((
                            previous_processed_wave[:-OVERLAP_SAMPLES],
                            blended_part
                        ))
                        
                        output_bytes = output_wave.astype(np.int16).tobytes()
                        conn.sendall(output_bytes)

                        previous_processed_wave = current_wave
                
                except socket.timeout:
                    idle_timeout_counter += 1
                    if idle_timeout_counter >= MAX_IDLE_TIMEOUTS:
                        print("アイドル状態が続いたため、接続を終了します。")
                        break
                    continue

    except ConnectionResetError:
        print(f"クライアント {addr} との接続が強制的に切断されました。")
    except Exception as e:
        print(f"クライアント {addr} との通信中に予期せぬエラーが発生しました: {e}")
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")


def start_server(args):
    """
    サーバーを起動し、クライアントの接続を待ち受けるメインループ。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(1.0)
        
        print(f"サーバーが {HOST}:{PORT} で待機中です。(停止するには Ctrl+C を押してください)")
        
        while True:
            try:
                conn, addr = s.accept()
                handle_client_connection(conn, addr)
                print("次の接続待機ループに戻ります...")

            except socket.timeout:
                continue
            except KeyboardInterrupt:
                print("\n停止信号を検知。メインループを抜けます。")
                break
            except Exception as e:
                print(f"サーバーのメインループでエラーが発生しました: {e}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="リアルタイム声質変換サーバー")
    parser.add_argument("--hpfile", type=str, default="FreeVC/configs/freevc-s.json", help="JSON設定ファイルへのパス")
    parser.add_argument("--ptfile", type=str, default="FreeVC/checkpoints/G_190000.pth", help="モデルチェックポイントファイルへのパス")
    parser.add_argument("--tgtwav", type=str, default="FreeVC/inputs/zundamon/recitation001.wav", help="目標話者のWAVファイルへのパス")
    args = parser.parse_args()

    print("声質変換モデルを初期化しています...")
    try:
        convert_rt.load_models(args.hpfile, args.ptfile, args.tgtwav)
    except FileNotFoundError as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        exit(1)
    
    start_server(args)
    print("サーバープログラムを終了します。")
