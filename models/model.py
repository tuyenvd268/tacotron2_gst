import torch.nn.functional as F
from math import sqrt
from utils import *
from models.encoder import Encoder
from models.decoder import Decoder
from models.reference_encoder import GST
from models.layers import *
import torch
import models.module as mm

from torch import nn

class TPSENet(nn.Module):
    """
    Predict speakers from style embedding (N-way classifier)

    """
    def __init__(self, text_dims, style_dims):
        super(TPSENet, self).__init__()
        self.conv = nn.Sequential(
            mm.Conv1d(text_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False),
        )
        self.gru = nn.GRU(style_dims, style_dims, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(style_dims*2, style_dims)
    
    def forward(self, text_embedding):
        """
        :param text_embedding: (N, Tx, E)

        Returns:
            :y_: (N, 1, n_speakers) Tensor.
        """
        te = text_embedding.transpose(1, 2) # (N, E, Tx)
        h = self.conv(te)
        h = h.transpose(1, 2) # (N, Tx, C)
        out, _ = self.gru(h)
        se = self.fc(out[:, -1:, :])
        se = torch.tanh(se)
        return se

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
        
        self.gst = GST(config)
        self.tpnet = TPSENet(text_dims=512, style_dims=256)
        self.ffw = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.1)
        

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[2].data.masked_fill_(mask, 0.0)
            outputs[3].data.masked_fill_(mask, 0.0)
            outputs[4].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)        
        
        emotion_embeddings = self.gst(mels.transpose(1, 2))    
        predicted_emotion_embeddings = self.tpnet(encoder_outputs)    

        embedded_emotions = emotion_embeddings.repeat(1, encoder_outputs.size(1), 1)
        encoder_outputs = torch.cat((encoder_outputs, embedded_emotions), dim=2)
        encoder_outputs = self.ffw(encoder_outputs)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [emotion_embeddings, predicted_emotion_embeddings, mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
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