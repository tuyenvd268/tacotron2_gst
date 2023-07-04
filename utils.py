import torch
import yaml
from scipy.io.wavfile import read
import numpy as np

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    if torch.cuda.is_available():
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def load_wavpath_and_text(path, split="|"):
    with open(path, "r", encoding="utf-8") as tmp:
        filepath_and_text = [line.strip().split(split) for line in tmp.readlines()]
    return filepath_and_text

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def load_yaml(path):
    with open(path, "r") as f:
        yml = yaml.safe_load(f)
    return yml

def parse_batch(batch):
    emotion_label, text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    emotion_label = to_gpu(emotion_label).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return (
        (text_padded, input_lengths, mel_padded, max_len, output_lengths),
        (emotion_label, mel_padded, gate_padded))