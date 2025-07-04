# client.py

import socket
import pyaudio
import threading
import time
import argparse
import sys

# --- 設定 ---
# ▼▼▼ このIPアドレスを、サーバーが動作しているIPアドレスに書き換えてください ▼▼▼
SERVER_IP = '192.168.101.106'  # 例: '10.66.28.18' や 'localhost'
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
    print("マイク音声の送信を開始します... (停止するには Ctrl+C を押してください)")
    while not stop_event.is_set():
        try:
            data = stream_in.read(CHUNK, exception_on_overflow=False)
            sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            print("サーバーとの接続が切れました。送信を停止します。")
            break
        except Exception as e:
            if not stop_event.is_set():
                print(f"送信中にエラーが発生しました: {e}")
            break

# --- サーバーからの受信と音声出力（スピーカー）を行う関数 ---
def receive_and_play(sock, stream_out, output_file):
    print("サーバーからの音声受信を開始します...")
    while not stop_event.is_set():
        try:
            response = sock.recv(CHUNK)
            if response:
                # 1. スピーカーで再生
                stream_out.write(response)
                # 2. ★★★ ファイルにバイナリデータを書き込む ★★★
                if output_file:
                    output_file.write(response)
            else:
                print("サーバーが接続を閉じました。受信を停止します。")
                break
        except (BrokenPipeError, ConnectionResetError):
            print("サーバーとの接続が切れました。受信を停止します。")
            break
        except socket.timeout:
            continue # タイムアウトは正常。ループを継続。
        except Exception as e:
            if not stop_event.is_set():
                print(f"受信中にエラーが発生しました: {e}")
            break
    print("受信スレッドを終了します。")

def main(args):
    stop_event.clear()
    p = pyaudio.PyAudio()
    stream_in = None
    stream_out = None
    s = None
    output_file = None # ファイルオブジェクトを初期化

    try:
        # ★★★ 出力ファイルを開く ★★★
        if args.output_file:
            try:
                output_file = open(args.output_file, 'wb')
                print(f"受信データを '{args.output_file}' に保存します。")
            except IOError as e:
                print(f"エラー: ファイル '{args.output_file}' を開けませんでした: {e}")
                return

        stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, 
                           frames_per_buffer=CHUNK, input_device_index=args.input_device)
        print(f"入力デバイス: {p.get_device_info_by_index(args.input_device)['name'] if args.input_device is not None else 'デフォルト'}")

        stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, 
                            frames_per_buffer=CHUNK, output_device_index=args.output_device)
        print(f"出力デバイス: {p.get_device_info_by_index(args.output_device)['name'] if args.output_device is not None else 'デフォルト'}")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0) # 受信スレッドのタイムアウトを設定
        s.connect((SERVER_IP, SERVER_PORT))
        print(f"サーバー {SERVER_IP}:{SERVER_PORT} に接続しました。")

        # ★★★ 受信スレッドにファイルオブジェクトを渡す ★★★
        thread_send = threading.Thread(target=send_mic_input, args=(s, stream_in))
        thread_receive = threading.Thread(target=receive_and_play, args=(s, stream_out, output_file))

        thread_send.start()
        thread_receive.start()

        while thread_send.is_alive() and thread_receive.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n停止信号を受信しました。プログラムを終了します...")
    except ConnectionRefusedError:
        print("サーバーに接続できませんでした。サーバーが起動しているか、IPアドレスとポートが正しいか確認してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        print("クリーンアップ処理を開始します...")
        stop_event.set()
        
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
            
        if 'thread_send' in locals() and thread_send.is_alive():
            thread_send.join()
        if 'thread_receive' in locals() and thread_receive.is_alive():
            thread_receive.join()
            
        # ★★★ ファイルを閉じる ★★★
        if output_file:
            output_file.close()
            print(f"受信データを '{args.output_file}' に保存しました。")
            
        if p:
            p.terminate()
        print("リソースを解放し、プログラムを終了しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="リアルタイム音声ストリーミングクライアント")
    parser.add_argument('--list-devices', action='store_true', help='利用可能なオーディオデバイスの一覧を表示して終了します。')
    parser.add_argument('-i', '--input-device', type=int, help='入力デバイスのインデックス番号。')
    parser.add_argument('-o', '--output-device', type=int, help='出力デバイスのインデックス番号。')
    # ★★★ 新しい引数を追加 ★★★
    parser.add_argument('--output-file', type=str, help='受信した音声を保存するバイナリファイル名。例: output.raw')
    args = parser.parse_args()

    if args.list_devices:
        p = pyaudio.PyAudio()
        list_audio_devices(p)
        p.terminate()
        sys.exit()
    
    main(args)
