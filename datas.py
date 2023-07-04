import torch
from torch import nn
from models.layers import TacotronSTFT
import utils
from torch.utils.data import Dataset
import numpy as np
from text import text_to_ids
import os

__all__ = ["TextMelDataset", "TextMelCollate"]

class TextMelDataset(Dataset):
    def __init__(self, wavpaths_and_texts, config) -> None:
        super().__init__()
        self.metadatas = utils.load_wavpath_and_text(wavpaths_and_texts)
        self.max_wav_value = config["max_wav_value"]
        self.sampling_rate = config["sampling_rate"]
        self.load_mel_from_disk = config["load_mel_from_disk"]
        self.emotion2id = config["emotion2id"]
        self.stft = TacotronSTFT(
            filter_length=config["filter_length"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            n_mel_channels=config["n_mel_channels"],
            sampling_rate=config["sampling_rate"],
            mel_fmax=config["mel_fmax"],
            mel_fmin=config["mel_fmin"]
        )
        self.wav_dir = config["wav_dir"]
    def get_text(self, text):
        text = text.lstrip("{").rstrip("}")
        text_norm = torch.IntTensor(
            text_to_ids(text)
        )
        return text_norm
    
    def get_mel(self, filename):
        path = os.path.join(self.wav_dir, filename+".npy")
        # audio, sampling_rate = utils.load_wav_to_torch(path)
        # if sampling_rate != self.stft.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(
        #         sampling_rate, self.stft.sampling_rate))
        # audio_norm = audio / self.max_wav_value
        # audio_norm = audio_norm.unsqueeze(0)
        # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # melspec = self.stft.mel_spectrogram(audio_norm)
        # melspec = torch.squeeze(melspec, 0)
        melspec = torch.from_numpy(np.load(path))

        return melspec
    
    def get_mel_text_pair(self, metadata):
        wav_id, phoneme, text, emotion = metadata[0], metadata[1], metadata[2], metadata[3]
        
        text = self.get_text(phoneme)
        mel = self.get_mel(wav_id)
        emotion_label = self.emotion2id[emotion]
        
        return (text, mel, emotion_label)
    def __getitem__(self, index):
        if len(self.metadatas[index]) == 0:
          return self.__getitem__(index-1)
      
        return self.get_mel_text_pair(self.metadatas[index])
    
    def __len__(self):
        return len(self.metadatas)

class TextMelCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        emotion_label = []
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            emotion_label.append(batch[ids_sorted_decreasing[i]][2])
            
        emotion_label = torch.Tensor(emotion_label)

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            
        return emotion_label, text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths