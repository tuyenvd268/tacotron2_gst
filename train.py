
from torch.utils.tensorboard import SummaryWriter
from datas import TextMelCollate, TextMelDataset
from datetime import datetime
from torch.utils.data import DataLoader
import statistics
from models.model import Tacotron2
from utils import *
from tqdm import tqdm
from loss import Tacotron2Loss
import numpy as np
import os

def prepare_dataloaders(config):
    trainset = TextMelDataset(config["training_files"], config)
    valset = TextMelDataset(config["validation_files"], config)
    collate_fn = TextMelCollate(config["n_frames_per_step"])

    train_sampler = None
    shuffle = True

    train_loader = DataLoader(
        trainset, 
        shuffle=shuffle,sampler=train_sampler,
        batch_size=config["batch_size"], pin_memory=True,
        num_workers=2,
        drop_last=True, collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valset, 
        shuffle=shuffle,
        batch_size=config["batch_size"], pin_memory=False,
        drop_last=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader

def init_model(config):
    model = Tacotron2(config)
    
    return model

def save_checkpoint(model, optimizer, step, path):
    torch.save(
        {
            "step":step,
            "mode_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict()
        }, 
        path
    )
    print(f"saved model and optimizer state dict at step {step} to {path}")
    
def load_checkpoint(path, model, optimizer):
    state_dict = torch.load(path, map_location="cpu")
    step = state_dict["step"]
    try:
        model.load_state_dict(state_dict["mode_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print(f"load checkpoint from {path} at step {step}.")
    except:
        current_model_dict = model.state_dict()
        loaded_state_dict = state_dict["mode_state_dict"]
        new_state_dict={
            k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Warning!!! Force loadding checkpoint from {path} at step {step}.")
    
    
    
    
    return model, optimizer, step

def init_logger_and_directories(config):
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    log_dir = f"{config['log_dir']}/{current_time}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"logging into {log_dir}")
    else:
        raise Exception("current log dir is exist !!!")

    writer = SummaryWriter(
        log_dir=log_dir)
    
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
        print(f'maker dir: {config["checkpoint_dir"]}')
    
    return writer

def train(cfg_path):
    config = load_yaml(cfg_path)
    writer = init_logger_and_directories(config)
    
    model = init_model(config=config)
    
    train_loader, val_loader = prepare_dataloaders(config)
    criterion = Tacotron2Loss()
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay
    )
    step = 1
    if os.path.exists(config["checkpoint"]):
        model, optimizer, step = load_checkpoint(config["checkpoint"], model, optimizer)
    
    model.train()
    for epoch in range(0, int(config["epoch"])):
        train_tqdm = tqdm(train_loader, desc=f"epoch: {epoch}")
        train_mel_losses, train_gate_losses, train_emotion_losses = [], [], []
        for batch in train_tqdm:
            model.zero_grad()
            optimizer.zero_grad()
            
            x, y = parse_batch(batch)
            y_pred = model(x)
            
            mel_loss, gate_loss, emotion_loss = criterion(y_pred, y)
            
            loss = mel_loss+gate_loss+emotion_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_thresh"])
            optimizer.step()
            
            train_mel_losses.append(mel_loss.item())
            train_gate_losses.append(gate_loss.item())
            train_emotion_losses.append(emotion_loss.item())
            
            train_tqdm.set_postfix({"mel_loss":mel_loss.item(), "emotion_loss":emotion_loss.item(),"gate_loss":gate_loss.item(), "step": step})
            step += 1
                    
            if step % int(config["save_checkpoint_per_steps"]) == 0:
                state_dict_path = f'{config["checkpoint_dir"]}/checkpoint_{step}.pt'
                save_checkpoint(model, optimizer, step, state_dict_path)
                
            if step % int(config["logging_per_steps"]) == 0:
                val_mel_loss, val_gate_loss, val_emotion_loss = validate(model, criterion, val_loader, step)
                train_mel_loss, train_gate_loss, train_emotion_loss = \
                    statistics.mean(train_mel_losses), statistics.mean(train_gate_losses), statistics.mean(train_emotion_losses)
                    
                writer.add_scalars('loss/mel', {"train": train_mel_loss, "val": val_mel_loss}, step)
                writer.add_scalars('loss/gate', {"train": train_gate_loss, "val": val_gate_loss}, step)
                writer.add_scalars('loss/emotion', {"train": train_emotion_loss, "val": val_emotion_loss}, step)

        print("###train loss at epoch={}: \nmel_loss={:4f} gate_loss={:4f} emotion_loss={:4f}\n" \
            .format(step, statistics.mean(train_mel_losses), statistics.mean(train_gate_losses), statistics.mean(train_emotion_losses)))
            
def validate(model, criterion, val_loader, step):
    model.eval()
    mel_losses, gate_losses, emotion_losses = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x, y = parse_batch(batch)
            y_pred = model(x)
            mel_loss, gate_loss, emotion_loss = criterion(y_pred, y)
            
            loss = mel_loss+gate_loss+emotion_loss
            
            mel_losses.append(mel_loss.item())
            gate_losses.append(gate_loss.item())
            emotion_losses.append(emotion_loss.item())
    model.train()
    
    print("###validation loss at step={}: \nmel_loss={:4f} gate_loss={:4f} emotion_loss={:4f}\n".format(step, statistics.mean(mel_losses), statistics.mean(gate_losses), statistics.mean(emotion_losses)))
    return (statistics.mean(mel_losses), statistics.mean(gate_losses), statistics.mean(emotion_losses))

if __name__ == "__main__":
    config_path = "config.yml"
    
    train(cfg_path=config_path)