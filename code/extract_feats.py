import argparse
import os
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq

from utils import init_torch_device

def get_arguments():
    parser = argparse.ArgumentParser(description="Extract wav2vec features from WAV files.")
    parser.add_argument('--wav_folder', '-w', type=str, required=True, help='Input directory with WAV files')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to wav2vec model checkpoint')
    parser.add_argument('--save_dir', '-s', type=str, required=True, help='Directory to save output .npy files')
    parser.add_argument('--feature', '-ft', type=str, default='c', help='Feature type to extract (z or c)')
    parser.add_argument('--cuda_device', '-d', type=int, default=0, help='CUDA device index')
    return parser.parse_args()

def prepare_model(checkpoint_path, device):
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    base_model = models[0]
    feature_extractor = import_fairseq_model(base_model)
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor

def extract_features_from_wav(wav_fp, extractor, device):
    waveform, _ = torchaudio.load(wav_fp)
    waveform = waveform.to(device)
    features, _ = extractor.extract_features(waveform)
    arr = features[0].squeeze(0).detach().cpu().numpy()
    if arr.shape[0] % 2 != 0:
        arr = arr[:-1]
    return arr

def main():
    args = get_arguments()
    wav_dir = Path(args.wav_folder)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = init_torch_device(args.cuda_device)
    extractor = prepare_model(args.checkpoint, device)

    wav_files = sorted(wav_dir.glob('*.wav'))
    print(f"Found {len(wav_files)} WAV files in {wav_dir}")

    for wav_file in tqdm(wav_files, desc="Extracting features"):
        feats = extract_features_from_wav(str(wav_file), extractor, device)
        output_fp = output_dir / (wav_file.stem + ".npy")
        np.save(output_fp, feats)

if __name__ == "__main__":
    main()
