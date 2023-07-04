import torch.nn.functional as F
from torch.autograd import Variable
from models.layers import *
import torch
from torch import nn
    
class PreNet(nn.Module):
    def __init__(self, in_dim, sizes) -> None:
        super().__init__()
        
        in_dims = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False) for in_size, out_size in zip(in_dims, sizes)]
        )
    
    def forward(self, input):
        for linear in self.layers:
            input = F.dropout(F.relu(linear(input)), p=0.5, training=True)
        return input

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim) -> None:
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            padding=padding,
            kernel_size=attention_kernel_size,
            bias=False,
            stride=1,
            dilation=1
        )
        
        self.location_dense = LinearNorm(
            in_dim=attention_n_filters,
            out_dim=attention_dim,
            bias=False,
            w_init_gain="tanh"
        )
        
    def forward(self, attention_weights_cat):
        """
        param:
            attention_weights_cat: shape=(B, 2, MAX_TIME)
        return:
            processed_attention: shape=(B, MAX_TIME, attention_dim)
        """
        
        processed_attention = self.location_conv(attention_weights_cat)
        # processed_attention: shape = (B, num_channels, MAX_TIME)
        processed_attention = processed_attention.transpose(1, 2)
        # processed_attention: shape = (B, MAX_TIME, num_channels)
        processed_attention = self.location_dense(processed_attention)
        # processed_attention: shape = (B, MAX_TIME, attention_dim)
        
        return processed_attention
        
class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_location_n_filter, attention_location_kernel_size) -> None:
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            in_dim=attention_rnn_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        self.memory_layer = LinearNorm(
            in_dim=embedding_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        
        self.v = LinearNorm(
            in_dim=attention_dim,
            out_dim=1,
            bias=False
        )
        
        self.location_layer = LocationLayer(
            attention_n_filters=attention_location_n_filter,
            attention_dim=attention_dim,
            attention_kernel_size=attention_location_kernel_size
        )
        
        self.score_mask_value = -float('inf')
        
    def get_alignment_energies(self, query, processed_memory,
                                attention_weights_cat):
        """
        param:
            query: shape = (B, n_mel_channels * n_frames_per_step)
            processed_memory: shape = (B, T_in, attention_dim)
            attention_weight_cat: shape = (B, 2, MAX_TIME)
        return:
            alignment: shape = (batch, MAX_TIME)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        # query: shape = (B, 1, n_mel_channels * n_frames_per_step)
        # processed_query: shape=(B, 1, attention_dim)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        # processed_attention_weights:  shape=(B, MAX_TIME, attention_dim)
        energies = self.v(torch.tanh(
            processed_memory + processed_query + processed_attention_weights
        ))
        
        # energies: shape=(B, MAX_TIME, 1)
        energies = energies.squeeze(-1)
        # energies: shape=(B, MAX_TIME)
        
        return energies
    
    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weight_cat, mask):
        """

        Args:
            attention_hidden_state (tensor): attention rnn last output
                shape=(B, 1, n_mel_channels * n_frames_per_step)
                
            memory (tensor): encoder output
                shape=(B, embedding_dim)
                
            processed_memory (tensor): processed encoder outputs
                shape=(B, 1, attention_dim)
                
            attention_weight_cat (tensor): previous and cummulative attention weights
                shape=(B, 2, MAX_TIME)
                
            mask (tensor): binary mask for padded data 
        """
        
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weight_cat
        )
        
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
            
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights
        
    
class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self._n_mel_channels = config["n_mel_channels"]
        self._n_frames_per_step = config["n_frame_per_step"]
        self._encoder_embedding_dim = config["encoder_embedding_dim"]
        self._attention_rnn_dim = config["attention_rnn_dim"]
        self._decoder_rnn_dim = config["decoder_rnn_dim"]
        self._prenet_dim = config["prenet_dim"]
        self._max_decoder_steps = config["max_decoder_steps"]
        self._p_attention_dropout = config["p_attention_dropout"]
        self._p_decoder_dropout = config["p_decoder_dropout"]
        self._attention_dim = config["attention_dim"]
        self._attention_location_n_filters = config["attention_location_n_filters"]
        self._attention_location_kernel_size = config["attention_location_kernel_size"]
        self._gate_threshold = config["gate_threshold"]
        
        self.prenet = PreNet(
            self._n_mel_channels * self._n_frames_per_step,
            [self._prenet_dim, self._prenet_dim]
        )
        
        self.attention_rnn = nn.LSTMCell(
            self._prenet_dim + self._encoder_embedding_dim,
            self._attention_rnn_dim
        )
        
        self.attention_layer = Attention(
            attention_dim=self._attention_dim,
            embedding_dim=self._encoder_embedding_dim,
            attention_location_kernel_size=self._attention_location_kernel_size,
            attention_location_n_filter=self._attention_location_n_filters,
            attention_rnn_dim=self._attention_rnn_dim
        )
        
        self.decoder_rnn = nn.LSTMCell(
            input_size=self._attention_rnn_dim + self._encoder_embedding_dim,
            hidden_size=self._decoder_rnn_dim, bias=1
        )
        
        self.linear_projection = LinearNorm(
            in_dim=self._decoder_rnn_dim + self._encoder_embedding_dim,
            out_dim=self._n_mel_channels
        )
        
        self.gate_layer = LinearNorm(
            in_dim=self._decoder_rnn_dim + self._encoder_embedding_dim,
            out_dim=1,
            bias=True,
            w_init_gain="sigmoid"
        )
        
    def get_mask_from_lengths(self, lengths):
        """ get masking from lengths

        Args:
            lengths (tensor): shape=[B]
        """
        max_len = torch.max(lengths).item()
        if torch.cuda.is_available():
            ids = torch.arange(0, max_len).cuda()
        else:
            ids = torch.arange(0, max_len)
        mask = (ids < lengths.unsqueeze(1)).bool()
        
        return mask
    
    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self._n_mel_channels * self._n_frames_per_step).zero_())
        return decoder_input
    
    def parse_decoder_inputs(self, decoder_inputs):
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self._n_frames_per_step), -1
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        
        return decoder_inputs
    
    def initialize_decoder_states(self, memory, mask):
        """init some term

        Args:
            memory (tensor): shape=[B, MAX_TIME, encoder_embedding_dim]
            mask (tensor): shape=[B, MAX_TIME]
        """
        
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        
        self.attention_hidden = Variable(memory.data.new(
            B, self._attention_rnn_dim).zero_())
        
        self.attention_cell = Variable(memory.data.new(
            B, self._attention_rnn_dim).zero_())
        
        self.decoder_hidden = Variable(memory.data.new(
            B, self._decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self._decoder_rnn_dim).zero_())

        
        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self._encoder_embedding_dim).zero_())
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        
    def decode(self, decoder_input):
        cell_input = torch.cat(
            (decoder_input, self.attention_context), -1
        )
        
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        
        self.attention_hidden = F.dropout(self.attention_hidden, self._p_attention_dropout, self.training)
        
        attention_weight_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim = 1
        )        
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weight_cat, self.mask
        )
        
        
        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1
        )
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self._p_decoder_dropout, self.training
        )
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )
        
        gate_prediction = self.gate_layer(
            decoder_hidden_attention_context
        )
        
        return decoder_output, gate_prediction, self.attention_weights
        
    def forward(self, memory, decoder_inputs, memory_lengths):
        """forward

        Args:
            memory (tensor): encoder outputs
                shape=[B, num_symbols, embedding_dim]
            decoder_inputs (tensor): decoder inputs 
            memory_lengths (tensor): encoder output lengths for attention masking
            
        return:
            mel_outputs: mel outputs from the decoder
            gate_output: gate outputs from decoder
            alignments: sequence of attention weight from the decoder
        """
        
        # shape=[B, encode_embedding]
        decoder_input = self.get_go_frame(memory=memory).unsqueeze(0)
        # decoder_input: shape=[1, B, n_mel_channels]
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs=decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        # decoder_inputs: shape=[T_out+1, B, n_mel_channels]
        decoder_inputs = self.prenet(decoder_inputs)
        # decoder_inputs: shape=[T_out+1, B, decoder_embedding_dim]
        
        self.initialize_decoder_states(
            memory, mask=~self.get_mask_from_lengths(memory_lengths)
        )
        
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input=decoder_input
            )
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
    
    
    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self._n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments
    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self._gate_threshold:
                break
            elif len(mel_outputs) == self._max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments