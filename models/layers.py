import torch
from torch import nn
import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=None, bias=True, w_init_gain="linear",
                 dilation=1) -> None:
        super(ConvNorm, self).__init__()
        
        if padding is None:
            padding = int(dilation * (kernel_size - 1)/2)
            
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        
        nn.init.xavier_uniform_(self.conv_layer.weight)
        
    def forward(self, input):
        output = self.conv_layer(input)
        
        return output
    
class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias= True, w_init_gain='linear') -> None:
        super().__init__()
        
        self.linear_layer=nn.Linear(in_dim, out_dim)
        
        nn.init.xavier_uniform_(self.linear_layer.weight)
        
    def forward(self, input):
        output = self.linear_layer(input)
        
        return output
    
class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,n_fft= filter_length, n_mels=n_mel_channels,fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output