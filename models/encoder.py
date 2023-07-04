import torch
import yaml
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder_kernel_size = config["encoder_kernel_size"]
        self.encoder_num_convolutions = config["encoder_n_convolutions"]
        self.encoder_embedding_dim = config["encoder_embedding_dim"]
        self.symbols_embedding_dim = config["symbols_embedding_dim"]
        
        # layers
        convs = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.encoder_embedding_dim,
                    out_channels=self.encoder_embedding_dim,
                    kernel_size=self.encoder_kernel_size,
                    padding=int((self.encoder_kernel_size-1)/2.0),
                    stride=1,
                    dilation=1
                ),
                nn.BatchNorm1d(self.encoder_embedding_dim)
            )
        ]
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(
            self.encoder_embedding_dim,
            int(self.encoder_embedding_dim/2),
            1,
            bidirectional=True,
            batch_first=True,
        )
        
    def forward(self, input, input_lengths):
        '''
        param:
            input(tensor): shape=[batch_size, embedding_dim, num_symbols]
        return:
            output(tensor): shape=[batch_size, num_symbols, embedding_dim]
        '''
        for conv_layer in self.convs:
            input = F.dropout(F.relu(conv_layer(input)), 0.5)
        
        input = input.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()

        output = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(output)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output
    
    def inference(self, x):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
        
def load_yaml(path):
    with open(path, "r") as f:
        yml = yaml.safe_load(f)
    return yml

if __name__ == "__main__":
    cfg_path = 'project3/configs/config.yml'
    config = load_yaml(cfg_path)
    print(config["model"])
    
    encoder = Encoder(config["model"]["encoder"])
    
    x = torch.randn(8, 512, 16)
    input_lengths = torch.tensor([i for i in range(1, 9)])
    
    output = encoder(x, input_lengths)
    print(output.shape)