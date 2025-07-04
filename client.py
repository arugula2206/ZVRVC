# client.py

import socket
import pyaudio
import threading
import time
import argparse
import sys

# --- 設定 ---
# ▼▼▼ このIPアドレスを、サーバーが動作しているIPアドレスに書き換えてください ▼▼▼
SERVER_IP = 'localhost'  # サーバーのIPアドレス (サーバー側と合わせる)
# ▲▲▲ ここまで ▲▲▲

SERVER_PORT = 8080          # サーバーのポート番号 (サーバー側と合わせる)
FORMAT = pyaudio.paInt16    # 音声フォーマット (16ビット整数)
CHANNELS = 1                # モノラル
RATE = 16000                # サンプリングレート (サーバーのモデル要件に合わせる)
CHUNK = 4096                # 一度に処理するデータサイズ (サーバー側と合わせる)

# --- スレッドを停止させるための共有フラグ ---
stop_event = threading.Event()

def list_audio_devices(p):
    """
    利用可能なオーディオデバイスの一覧を表示する関数。
    """
    print("-" * 40)
    print("利用可能なオーディオデバイス:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        device_type = ""
        if info.get('maxInputChannels') > 0:
            device_type += "[入力]"
        if info.get('maxOutputChannels') > 0:
            device_type += "[出力]"
        
        print(f"  インデックス {info['index']}: {info['name']} {device_type}")
    print("-" * 40)

# --- 音声入力（マイク）とサーバーへの送信を行う関数 ---
def send_mic_input(sock, stream_in):
    print("[送信スレッド] 開始... (停止するには Ctrl+C を押してください)")
    chunk_counter = 0
    while not stop_event.is_set():
        try:
            # print(f"[送信スレッド] ({chunk_counter}) マイクからデータ読み取り中...") # 詳細すぎるのでコメントアウト
            data = stream_in.read(CHUNK, exception_on_overflow=False)
            # print(f"[送信スレッド] ({chunk_counter}) {len(data)}バイト読み取り完了。") # 詳細すぎるのでコメントアウト
            sock.sendall(data)
            # print(f"[送信スレッド] ({chunk_counter}) {len(data)}バイト送信完了。") # 詳細すぎるのでコメントアウト
            chunk_counter += 1
        except (BrokenPipeError, ConnectionResetError):
            print("[送信スレッド] サーバーとの接続が切れました。")
            break
        except Exception as e:
            if not stop_event.is_set():
                print(f"[送信スレッド] エラーが発生しました: {e}")
            break
    print("[送信スレッド] 終了。")

# --- サーバーからの受信と音声出力（スピーカー）を行う関数 ---
def receive_and_play(sock, stream_out):
    print("[受信スレッド] 開始...")
    chunk_counter = 0
    while not stop_event.is_set():
        try:
            # print(f"[受信スレッド] ({chunk_counter}) サーバーからのデータ受信待機中...") # 詳細すぎるのでコメントアウト
            response = sock.recv(CHUNK)
            if response:
                # print(f"[受信スレッド] ({chunk_counter}) {len(response)}バイト受信完了。") # 詳細すぎるのでコメントアウト
                stream_out.write(response)
                # print(f"[受信スレッド] ({chunk_counter}) スピーカーへ書き込み完了。") # 詳細すぎるのでコメントアウト
                chunk_counter += 1
            else:
                print("[受信スレッド] サーバーが接続を閉じました。")
                break
        except (BrokenPipeError, ConnectionResetError):
            print("[受信スレッド] サーバーとの接続が切れました。")
            break
        except Exception as e:
            if not stop_event.is_set():
                print(f"[受信スレッド] エラーが発生しました: {e}")
            break
    print("[受信スレッド] 終了。")

def main(args):
    print("[メイン] プログラム開始。")
    stop_event.clear()

    p = pyaudio.PyAudio()
    print("[メイン] PyAudio初期化完了。")
    stream_in = None
    stream_out = None
    s = None

    try:
        print("[メイン] 入力ストリームを開いています...")
        stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                           frames_per_buffer=CHUNK, input_device_index=args.input_device)
        print(f"  -> 入力デバイス: {p.get_device_info_by_index(args.input_device)['name'] if args.input_device is not None else 'デフォルト'}")

        print("[メイン] 出力ストリームを開いています...")
        stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, 
                            frames_per_buffer=CHUNK, output_device_index=args.output_device)
        print(f"  -> 出力デバイス: {p.get_device_info_by_index(args.output_device)['name'] if args.output_device is not None else 'デフォルト'}")

        print(f"[メイン] サーバー {SERVER_IP}:{SERVER_PORT} へ接続を試みています...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_IP, SERVER_PORT))
        print("[メイン] サーバーへ接続しました。")

        print("[メイン] 送受信スレッドを作成・開始します...")
        thread_send = threading.Thread(target=send_mic_input, args=(s, stream_in))
        thread_receive = threading.Thread(target=receive_and_play, args=(s, stream_out))

        thread_send.start()
        thread_receive.start()
        print("[メイン] スレッドを開始しました。メインスレッドは待機状態に入ります。")

        while thread_send.is_alive() and thread_receive.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[メイン] Ctrl+Cを検知しました。終了処理を開始します...")
    except ConnectionRefusedError:
        print("[メイン] サーバーに接続できませんでした。サーバーが起動しているか、IPアドレスとポートが正しいか確認してください。")
    except Exception as e:
        print(f"[メイン] エラーが発生しました: {e}")
    finally:
        print("[メイン] クリーンアップ処理を開始します...")
        
        print("  -> 1. 停止イベントをセットします...")
        stop_event.set()
        
        print("  -> 2. ソケットとストリームを閉じます...")
        if s:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass 
            s.close()
        if stream_in:
            stream_in.stop_stream()
            stream_in.close()
        if stream_out:
            stream_out.stop_stream()
            stream_out.close()
            
        print("  -> 3. スレッドの終了を待ちます...")
        if 'thread_send' in locals() and thread_send.is_alive():
            thread_send.join()
        if 'thread_receive' in locals() and thread_receive.is_alive():
            thread_receive.join()
            
        print("  -> 4. PyAudioを終了します...")
        if p:
            p.terminate()
        print("[メイン] リソースを解放し、プログラムを終了しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="リアルタイム音声ストリーミングクライアント")
    parser.add_argument('--list-devices', action='store_true', help='利用可能なオーディオデバイスの一覧を表示して終了します。')
    parser.add_argument('-i', '--input-device', type=int, help='入力デバイスのインデックス番号。')
    parser.add_argument('-o', '--output-device', type=int, help='出力デバイスのインデックス番号。')
    args = parser.parse_args()

    if args.list_devices:
        p = pyaudio.PyAudio()
        list_audio_devices(p)
        p.terminate()
        sys.exit()
    
    main(args)
