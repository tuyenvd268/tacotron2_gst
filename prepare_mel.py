import torch
from utils import load_wav_to_torch
from models.layers import TacotronSTFT
from utils import load_yaml
import numpy as np
from glob import glob
from tqdm import tqdm
import os

cfg_path = "config.yml"
config = load_yaml(cfg_path)

stft = TacotronSTFT(
            filter_length=config["filter_length"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            n_mel_channels=config["n_mel_channels"],
            sampling_rate=config["sampling_rate"],
            mel_fmax=config["mel_fmax"],
            mel_fmin=config["mel_fmin"]
        )
max_wav_value = 32768.0

wav_dir = "data/merged/wavs/*.wav"
mel_dir = "data/mels"
for path in tqdm(glob(wav_dir)):
    audio, sampling_rate = load_wav_to_torch(path)
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).numpy()
    
    outpath = f'{mel_dir}/{os.path.basename(path).replace(".wav", ".npy")}'
    np.save(outpath, melspec)