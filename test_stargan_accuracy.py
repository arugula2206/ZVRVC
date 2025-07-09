import argparse
import librosa
import numpy as np
import torch
import wave
from munch import Munch
import json

# ご提示いただいた変換コードをインポート
import starganv2_vc.inference_rt as converter

# HiFi-GAN（ボコーダー）関連のモジュールをインポート
from hifigan_fix.models import Generator as Hifigan
from hifigan_fix.meldataset import mel_spectrogram

def main(args):
    """
    ローカルファイルでStarGANv2-vcの音声変換をテストする
    """
    print("--- StarGANv2-vc 精度テストを開始します ---")

    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 2. HiFi-GAN（ボコーダー）の初期化
    print("HiFi-GAN（ボコーダー）を読み込んでいます...")
    with open(args.hifigan_config, 'r') as f:
        h = Munch(json.load(f))
    hifigan = Hifigan(h).to(device)
    hifigan.load_state_dict(torch.load(args.hifigan_model, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    print("HiFi-GANの読み込みが完了しました。")

    # 3. StarGANv2-vcの初期化（ご提示のコードを利用）
    print("\nStarGANv2-vcモデルを読み込んでいます...")
    converter.initialize_vc(h, device, args.stargan_model_dir, args.stargan_model_name, args.f0_model, 'model')
    print("StarGANv2-vcの読み込みが完了しました。")

    # 4. 入力音声の読み込みと前処理
    print(f"\n入力ファイル '{args.input}' を読み込んでいます...")
    input_wav, sr = librosa.load(args.input, sr=h.sampling_rate)
    input_wav_tensor = torch.from_numpy(input_wav).unsqueeze(0).to(device)
    print("入力ファイルの読み込みが完了しました。")

    # 5. 入力音声からメルスペクトログラムを生成
    with torch.no_grad():
        input_mel = mel_spectrogram(input_wav_tensor, h.n_fft, h.num_mels, h.sampling_rate,
                                    h.hop_size, h.win_size, h.fmin, h.fmax)

    # 6. 音声変換の実行（ご提示のコードを利用）
    print(f"\n音声変換を実行しています... (ターゲット: {args.speaker})")
    with torch.no_grad():
        converted_mel = converter.conversion(input_mel, args.speaker)
        print("メルスペクトログラムの変換が完了しました。")

        # 7. 変換後のメルスペクトログラムを音声波形に変換（HiFi-GANを使用）
        output_wav = hifigan(converted_mel).squeeze().cpu().numpy()
        print("音声波形への変換が完了しました。")

    # 8. 出力音声の保存
    output_wav_int16 = (output_wav * 32767.0).astype(np.int16)
    print(f"\n変換後の音声を '{args.output}' に保存しています...")
    with wave.open(args.output, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16bit
        wf.setframerate(h.sampling_rate)
        wf.writeframes(output_wav_int16.tobytes())
    
    print("\n--- テスト完了 ---")
    print(f"'{args.output}' を再生して変換精度を確認してください。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StarGANv2-vc 変換精度テスト")
    
    # モデルと設定ファイルのパスを引数として指定
    parser.add_argument('--stargan-model-dir', type=str, required=True, help='StarGANv2モデルが格納されているディレクトリ名')
    parser.add_argument('--stargan-model-name', type=str, required=True, help='StarGANv2モデルのファイル名 (.pth)')
    parser.add_argument('--f0-model', type=str, required=True, help='F0予測モデルのファイル名')
    parser.add_argument('--hifigan-config', type=str, required=True, help='HiFi-GANの設定ファイル (config.json) へのパス')
    parser.add_argument('--hifigan-model', type=str, required=True, help='HiFi-GANのモデルファイルへのパス')

    # 入出力ファイルと話者を指定
    parser.add_argument('-i', '--input', type=str, default='input.wav', help='入力WAVファイル')
    parser.add_argument('-o', '--output', type=str, default='output2.wav', help='変換後の出力WAVファイル')
    parser.add_argument('-s', '--speaker', type=str, default='zundamon127', help='目標話者のキー')
    
    args = parser.parse_args()
    main(args)