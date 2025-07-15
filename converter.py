# converter.py

import os
import librosa
import numpy as np
import torch
import yaml
import json
import soxr
from munch import Munch
import const as const

# 必要なモジュールをインポート
from hifigan_fix.meldataset import mel_spectrogram
from hifigan_fix.models import Generator as Hifigan
from starganv2_vc.Utils.JDC.model import JDCNet
from starganv2_vc.models import Generator, MappingNetwork, StyleEncoder
from frcrn import initialize_frcrn, denoise

# --- グローバル変数 ---
# このモジュール内でモデルや設定を保持するための変数
_device = None
_hps_hifigan = None
F0_model = None
starganv2 = None
hifigan = None
reference_embeddings = None
min_len_wave = 24000
use_denoiser = False # ノイズ除去機能が有効かどうかのフラグ
denoise_samplerate = 16000 # FRCRNが要求するサンプリングレート

def initialize_models(config):
    """
    サーバー起動時に一度だけ呼ばれ、全てのAIモデルを初期化する関数
    
    Args:
        config (dict): config.jsonから読み込まれた設定情報
    """
    global _device, _hps_hifigan, hifigan, use_denoiser
    
    # 1. 使用するデバイス（GPU/CPU）を決定
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {_device}")

    # 2. HiFi-GAN（ボコーダー）を初期化
    print("HiFi-GAN（ボコーダー）を読み込んでいます...")
    with open(config['hifigan_config'], 'r') as f:
        _hps_hifigan = Munch(json.load(f))
    hifigan = Hifigan(_hps_hifigan).to(_device)
    hifigan.load_state_dict(torch.load(config['hifigan_model'], map_location=_device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    print("HiFi-GANの読み込みが完了しました。")

    # 3. StarGANv2-vc本体を初期化
    _initialize_vc(
        _hps_hifigan, _device, 
        config['stargan_model_dir'], config['stargan_model_name'], 
        config['f0_model'], config['f0_model_key']
    )
    
    # 4. FRCRN（ノイズ除去）を初期化（設定が有効な場合のみ）
    use_denoiser = config.get('use_denoiser', False)
    if use_denoiser:
        print("FRCRNノイズ除去モデルを初期化しています...")
        # 発話単位の変換では入力長が可変なため、十分なサイズで初期化
        initialize_frcrn(_device, int(_hps_hifigan.sampling_rate * 10 / _hps_hifigan.sampling_rate * denoise_samplerate))
        print("FRCRNの初期化が完了しました。")

    print("全てのモデルの初期化が完了しました。")

def _initialize_vc(h, device, model_dir, model_name, f0_model_path, f0_model_key):
    """StarGANv2-vcの関連モデルを初期化する内部関数"""
    global F0_model, starganv2, reference_embeddings
    vc_dir_path = os.path.dirname(os.path.abspath(__file__))

    print("F0予測モデルを読み込んでいます...")
    F0_model = JDCNet(num_class=1, seq_len=192).to(device)
    full_f0_path = os.path.join(vc_dir_path, 'starganv2_vc', 'Utils', 'JDC', f0_model_path)
    params = torch.load(full_f0_path, weights_only=False)[f0_model_key]
    F0_model.load_state_dict(params)
    _ = F0_model.eval()

    print("StarGANv2モデルを読み込んでいます...")
    model_path = os.path.join(vc_dir_path, 'starganv2_vc', 'Models', model_dir, model_name)
    with open(os.path.join(vc_dir_path, 'starganv2_vc', 'Configs', 'config.yml')) as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config['model_params'])
    params = torch.load(model_path, map_location='cpu')['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval().to(device) for key in starganv2]

    print("参照話者のスタイル辞書を作成しています...")
    speaker_dicts = {}
    for s in const.speakers:
        for r in range(1, const.references + 1):
            path = os.path.join(vc_dir_path, 'starganv2_vc', 'Data', 'ITA-corpus', s, f'recitation{r:03}.wav')
            speaker_id = const.speakers.index(s) + 1
            speaker_dicts[f'{s}{r:03}'] = (path, speaker_id)
            
    reference_embeddings = _compute_style(speaker_dicts)
    print(f"スタイル辞書の作成が完了しました。{len(reference_embeddings.keys())}件の話者をロードしました。")

def build_model(model_params):
    """StarGANv2の各コンポーネントを構築する"""
    args = Munch(model_params)
    return Munch(
        generator=Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel),
        mapping_network=MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim),
        style_encoder=StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    )

def _compute_style(speaker_dicts):
    """参照音声ファイル群から、話者ごとの声質（スタイル）を抽出する"""
    local_ref_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        wave, sr = librosa.load(path, sr=24000, res_type='soxr_vhq')
        wave, _ = librosa.effects.trim(wave, top_db=30)
        if len(wave) < min_len_wave: wave = np.pad(wave, (0, min_len_wave - len(wave)))
        
        wave_tensor = torch.from_numpy(wave).float().unsqueeze(0).to(_device)
        mel_tensor = mel_spectrogram(
            wave_tensor, _hps_hifigan.n_fft, _hps_hifigan.num_mels, _hps_hifigan.sampling_rate,
            _hps_hifigan.hop_size, _hps_hifigan.win_size, _hps_hifigan.fmin, _hps_hifigan.fmax
        )
        with torch.no_grad():
            label = torch.LongTensor([speaker]).to(_device)
            ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        local_ref_embeddings[key] = (ref, label)
    return local_ref_embeddings

def _internal_conversion(mel, ref_emb_key):
    """メルスペクトログラムを変換する内部関数"""
    f0_feat = F0_model.get_feature_GAN(mel.unsqueeze(1))
    ref_tuple = reference_embeddings.get(ref_emb_key)
    if ref_tuple is None: raise ValueError(f"参照話者キー '{ref_emb_key}' が見つかりません。")
    
    out = starganv2.generator(mel.unsqueeze(1), ref_tuple[0], F0=f0_feat)
    return out.squeeze(1)

def convert_voice(audio_data_bytes, speaker_key):
    """
    音声バイトデータを受け取り、変換後の音声バイトデータを返す全工程（リサンプリング含む）
    """
    client_rate = 48000
    model_rate = _hps_hifigan.sampling_rate

    # 1. バイト -> float配列 (48kHz)
    audio_int16 = np.frombuffer(audio_data_bytes, dtype=np.int16)
    audio_float_48k = audio_int16.astype(np.float32) / 32768.0

    # 2. 48kHz -> 24kHz (モデルのレート) へリサンプリング
    audio_float_24k = soxr.resample(audio_float_48k, client_rate, model_rate, 'VHQ')
    
    # 3. (オプション) ノイズ除去
    if use_denoiser:
        print("ノイズ除去を実行しています...")
        # frcrnは16kHzを想定しているためリサンプリング
        audio_for_denoise = soxr.resample(audio_float_24k, model_rate, denoise_samplerate, 'VHQ')
        denoised_wave = denoise(audio_for_denoise)
        # 再びモデルのレートに戻す
        audio_float_24k = soxr.resample(denoised_wave, denoise_samplerate, model_rate, 'VHQ')

    input_wav_tensor = torch.from_numpy(audio_float_24k).unsqueeze(0).to(_device)

    with torch.no_grad():
        # 4. 音声 -> メルスペクトログラム (24kHz)
        input_mel = mel_spectrogram(
            input_wav_tensor, _hps_hifigan.n_fft, _hps_hifigan.num_mels, _hps_hifigan.sampling_rate,
            _hps_hifigan.hop_size, _hps_hifigan.win_size, _hps_hifigan.fmin, _hps_hifigan.fmax
        )
        
        # 5. メルスペクトログラムを声質変換
        converted_mel = _internal_conversion(input_mel, speaker_key)
        
        # 6. 変換後メルスペクトログラム -> 音声 (24kHz)
        output_wav_24k = hifigan(converted_mel)
    
    # 7. 24kHz -> 48kHz (クライアントのレート) へリサンプリング
    output_wav_24k_np = output_wav_24k.squeeze().cpu().numpy()
    output_wav_48k_np = soxr.resample(output_wav_24k_np, model_rate, client_rate, 'VHQ')

    # 8. float配列 -> バイト
    output_wav_int16 = (output_wav_48k_np * 32767.0).astype(np.int16)
    return output_wav_int16.tobytes()
