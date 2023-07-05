import torch
from scipy.io.wavfile import write
import numpy as np
import json
import os
import sys

from train import init_model
from text import text_to_ids, word_to_phoneme
from utils import load_yaml

from hifigan.models import Generator
from hifigan.env import AttrDict

MAX_WAV_VALUE = 32768.0

class Pipline():
    def __init__(self, config) -> None:
        encoder_config = load_yaml("config.yml")
        self.encoder_config=encoder_config
        self.encoder = init_model(encoder_config)
        
        self.encoder.load_state_dict(
            torch.load(config["checkpoint"], 
                        map_location=torch.device('cpu'))['mode_state_dict'])
    
        print(f'load: {config["checkpoint"]}')
        
        self.vocoder_config = self.load_vocoder_config("hifigan/config.json")
        self.vocoder = Generator(self.vocoder_config)
        self.vocoder.load_state_dict(
            torch.load(
                "outputs/checkpoints/generator_universal.pth.tar",
                map_location=torch.device('cpu'))['generator']
            )
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        
    def infer(self, text):
        normed_text, phoneme_ids, phoneme = self.prepare_input_for_infer(text)
        mel_spec, mel_outputs_postnet, alignments= self.infer_mel_spec(phoneme_ids)        
        output_path = self.infer_wav(mel_spec)

        return output_path, text, normed_text, phoneme, mel_spec, mel_outputs_postnet, alignments
    
    def prepare_input_for_infer(self, text):
        phoneme = word_to_phoneme(text)
        print(f'phoneme: {phoneme}')
        phoneme_ids = np.array(text_to_ids(phoneme))[None, :]
        
        return "normed_text", phoneme_ids, phoneme
        
    def load_vocoder_config(self, config_path):
        with open(config_path) as f:
            data = f.read()

        json_config = json.loads(data)
        config = AttrDict(json_config)
        
        return config
    
    @torch.no_grad()
    def infer_wav(self, mel_spec):
        x = torch.FloatTensor(mel_spec)
        y_g_hat = self.vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        output_file = 'outputs/test_generated_e2e.npy'
        write(output_file, self.vocoder_config.sampling_rate, audio)
        print(f'saved: {output_file}')
        
        return output_file
        
    def infer_mel_spec(self, phoneme, emotion2weight=0):
        emotion2weight = {
            "angry": 0.1,
            "happy": 0.1,
            "sad": 0.3,
            "neutral": 0.5
        }
        phoneme = torch.autograd.Variable(torch.from_numpy(phoneme)).long()
     
        emotion_weight=[0, 0, 0, 0]
        for i in range(len(emotion_weight)):
            emotion_weight[i] = emotion2weight[self.encoder_config["id2emotion"][i]]
        
        emotion_weight = torch.tensor(emotion_weight).unsqueeze(0)
        
        inputs = (phoneme, emotion_weight)
        mel_outputs, mel_outputs_postnet, _, alignments = self.encoder.inference(inputs)
        
        return mel_outputs.squeeze(0).detach().numpy(), mel_outputs_postnet.squeeze(0).detach().numpy(), alignments.squeeze(0).detach().numpy()