import torch.nn.functional as F
from math import sqrt
from utils import *
from models.encoder import Encoder
from models.decoder import Decoder
from models.reference_encoder import Reference_Encoder
from models.layers import *
import torch
from torch import nn

class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()
        self.mask_padding = config["mask_padding"]
        self.n_mel_channels = config["n_mel_channels"]
        self.n_frames_per_step = config["n_frames_per_step"]
        self.embedding = nn.Embedding(
            config["n_symbols"], 
            config["symbols_embedding_dim"]
        ).to(config['device'])
        std = sqrt(2.0 / (config["n_symbols"] + config["symbols_embedding_dim"]))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(config).to(config['device'])
        self.decoder = Decoder(config).to(config['device'])
        self.postnet = Postnet(config).to(config['device'])
        self.reference_encoder = Reference_Encoder(config)
        self.ffw = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.2)
        
        self.emotion_embeddings = nn.Parameter(
            torch.randn(4, 256)
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask, 0.0)
            outputs[3].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)        
        
        emotion_embeddings = self.reference_encoder(mels.transpose(1, 2))
        attention_weight = torch.matmul(emotion_embeddings, self.emotion_embeddings.T)
        emotion_predictions = attention_weight
        embedded_emotions = torch.matmul(
            torch.nn.functional.softmax(attention_weight, dim=1), self.emotion_embeddings).unsqueeze(1)
        
        embedded_emotions = embedded_emotions.repeat(1, encoder_outputs.size(1), 1)
        embedded_emotions = self.dropout(embedded_emotions)
        encoder_outputs = torch.cat((encoder_outputs, embedded_emotions), dim=2)
        encoder_outputs = self.ffw(encoder_outputs)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [emotion_predictions, mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        text, emotion_weight = inputs
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        
        print("emotion_weight: ", emotion_weight)
        embedded_emotions = torch.matmul(emotion_weight, self.emotion_embeddings).unsqueeze(1)
        embedded_emotions = embedded_emotions.repeat(1, encoder_outputs.size(1), 1)
        encoder_outputs = torch.cat((encoder_outputs, embedded_emotions), dim=2)
        encoder_outputs = self.ffw(encoder_outputs)
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

class Postnet(nn.Module):
    def __init__(self, config):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config["n_mel_channels"], config["postnet_embedding_dim"],
                         kernel_size=config["postnet_kernel_size"], stride=1,
                         padding=int((config["postnet_kernel_size"] - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(config["postnet_embedding_dim"]))
        )

        for i in range(1, config["postnet_n_convolutions"] - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(config["postnet_embedding_dim"],
                             config["postnet_embedding_dim"],
                             kernel_size=config["postnet_kernel_size"], stride=1,
                             padding=int((config["postnet_kernel_size"] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(config["postnet_embedding_dim"]))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config["postnet_embedding_dim"], config["n_mel_channels"],
                         kernel_size=config["postnet_kernel_size"], stride=1,
                         padding=int((config["postnet_kernel_size"] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(config["n_mel_channels"]))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x