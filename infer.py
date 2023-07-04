import matplotlib
import matplotlib.pylab as plt
import IPython.display as ipd
import numpy as np
import torch

from models import model
from models.layers import TacotronSTFT, STFT
from train import init_model
from text import text_to_ids
from utils import load_yaml

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
        
def infer(model, text):    
    sequence = np.array(text_to_ids(text))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).long()
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    
    return mel_outputs.squeeze(0).detach().numpy()


if __name__ == "__main__":
    checkpoint_path = "/home/tuyendv/Desktop/project3/checkpoints/tacotron2/checkpoint_36000.pt"
    cfg_path = 'configs/config.yml'
    res_path = "/home/tuyendv/Desktop/project3/inputs/test.npy"
    
    config = load_yaml(cfg_path)
    model = init_model(config)
    model.load_state_dict(
        torch.load(
            checkpoint_path, 
            map_location=torch.device('cpu')
        )['mode_state_dict'])
    
    text = "aa1 iz1 a1 iz1 <spc> k u5 ngz <spc> k o3 <spc> m oo6 tc <spc> h i2 nhz <spc> d u1 ngz <spc> ch u1 ngz <spc> v ee2 <spc> k a3 iz3 <spc> v i6 nhz <spc> h a6 <spc> l o1 ngz <spc> dd a5 <spc> th aa1 nc <spc> k w e1 nc <spc> dd ee3 nc <spc> m uw3 kc <spc> k o3 <spc> ph aa2 nc <spc> k u5 <spc> k i5 <spc> ."
    sequence = np.array(text_to_ids(text))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).long()
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    
    np.save(res_path, mel_outputs.squeeze(0).detach().numpy())