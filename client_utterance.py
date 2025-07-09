# client_modified.py (再生スキップ対応版)

import socket
import pyaudio
import time
import argparse
import sys
import numpy as np
import struct

# --- 設定 ---
SERVER_IP = 'localhost'
SERVER_PORT = 8080
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

# --- 発話検出（VAD）設定 ---
VAD_THRESHOLD = 300
SILENCE_CHUNKS = int(1.0 * RATE / CHUNK)
MAX_RECORD_CHUNKS = int(10 * RATE / CHUNK)

def list_audio_devices(p):
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

def main(args):
    p = pyaudio.PyAudio()
    stream_in = None
    stream_out = None

    try:
        stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                           frames_per_buffer=CHUNK, input_device_index=args.input_device)
        print(f"入力デバイス: {p.get_device_info_by_index(args.input_device)['name'] if args.input_device is not None else 'デフォルト'}")

        stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                            frames_per_buffer=CHUNK, output_device_index=args.output_device)
        print(f"出力デバイス: {p.get_device_info_by_index(args.output_device)['name'] if args.output_device is not None else 'デフォルト'}")

        print("\nクライアント起動完了。Ctrl+Cで終了します。")

        while True:
            print("\n-----------------------------------------")
            print("🎤 発話の開始を待っています...")
            while True:
                data = stream_in.read(CHUNK, exception_on_overflow=False)
                rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16).astype(np.float64))))
                if rms > VAD_THRESHOLD:
                    break
            
            print("🔥 発話を検知しました！ 録音中...")
            frames = [data]
            silent_count = 0
            while True:
                data = stream_in.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                rms = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16).astype(np.float64))))
                
                if rms < VAD_THRESHOLD:
                    silent_count += 1
                else:
                    silent_count = 0
                
                if silent_count > SILENCE_CHUNKS or len(frames) > MAX_RECORD_CHUNKS:
                    break
            
            recorded_data = b''.join(frames)
            print(f"💬 録音終了 ({len(recorded_data) / (RATE * 2):.2f}秒)。サーバーに接続して変換します...")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_IP, SERVER_PORT))
                s.sendall(struct.pack('>I', len(recorded_data)))
                s.sendall(recorded_data)
                print("🔊 音声データをサーバーに送信しました。")
                
                # ▼▼▼ 変更点: サーバーからの応答をチェック ▼▼▼
                response_len_data = s.recv(4)
                if not response_len_data:
                    raise ConnectionError("サーバーからデータ長の応答がありませんでした。")
                response_len = struct.unpack('>I', response_len_data)[0]

                # データ長が0より大きい場合のみ、データを受信して再生
                if response_len > 0:
                    print(f"✅ 変換済みデータ({response_len}バイト)を受信します。")
                    converted_data = b''
                    while len(converted_data) < response_len:
                        packet = s.recv(4096)
                        if not packet:
                            break
                        converted_data += packet
                    
                    print("🎶 変換後の音声を再生します...")
                    stream_out.write(converted_data)
                else:
                    # データ長が0の場合は、何もせず次のループへ
                    print("🔇 サーバーから再生不要の信号を受信しました。次の発話に移ります。")
                # ▲▲▲ 変更点 ここまで ▲▲▲
            
    except KeyboardInterrupt:
        print("\n終了します。")
    except ConnectionRefusedError:
        print("\n[エラー] サーバーに接続できませんでした。サーバーが起動しているか確認してください。")
    except Exception as e:
        print(f"\n[エラー] 予期せぬエラーが発生しました: {e}")
    finally:
        if stream_in:
            stream_in.stop_stream()
            stream_in.close()
        if stream_out:
            stream_out.stop_stream()
            stream_out.close()
        if p:
            p.terminate()
        print("リソースを解放しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="発話単位で音声を変換するクライアント")
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