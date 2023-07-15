
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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed():
    dist_url = "env://"
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def prepare_dataloaders(config):
    trainset = TextMelDataset(config["training_files"], config)
    valset = TextMelDataset(config["validation_files"], config)
    collate_fn = TextMelCollate(config["n_frames_per_step"])

    train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
    val_sampler = DistributedSampler(dataset=valset, shuffle=False)

    train_loader = DataLoader(
        trainset, 
        shuffle=False,sampler=train_sampler,
        batch_size=config["batch_size"], pin_memory=False,
        num_workers=1,
        drop_last=True, collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valset, 
        shuffle=False,sampler=val_sampler,
        num_workers=0,
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
        "mode_state_dict":model.module.state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
        }, 
        path
    )
    print(f"saved model and optimizer state dict at step {step} to {path}")
    
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

def train(config):
    init_distributed()
    
    if is_main_process():
        writer = init_logger_and_directories(config)
    train_loader, val_loader = prepare_dataloaders(config)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = init_model(config=config).cuda()

    criterion = Tacotron2Loss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    
    step = 1
    if os.path.exists(config["checkpoint"]):
        state_dict = torch.load(config["checkpoint"])
        step = state_dict["step"]
        try:
            model.load_state_dict(state_dict["mode_state_dict"])
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            
            print(f'load checkpoint from {config["checkpoint"]}')
        except:
            current_model_dict = model.state_dict()
            loaded_state_dict = state_dict["mode_state_dict"]
            new_state_dict={
                k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
            model.load_state_dict(new_state_dict, strict=False)
            print(f'Warning!!! Force load checkpoint from {config["checkpoint"]}')
        
    if step > config["reference_step"]:
        layers = [model.reference_encoder,]
        for module in layers:
            for param in module.parameters():
                param.requires_grad = False
                
        model.emotion_embeddings.requires_grad = False
        print("freezing reference encoder")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(0, int(config["epoch"])):
        # train_tqdm = tqdm(train_loader, desc=f"epoch: {epoch}")
        train_mel_losses, train_gate_losses, train_emotion_losses = [], [], []
        for batch in train_loader:
            model.zero_grad()
            
            x, y = parse_batch(batch)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                y_pred = model(x)
                mel_loss, gate_loss, emotion_loss = criterion(y_pred, y)
        
                loss = mel_loss+gate_loss+0.1*emotion_loss
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_thresh"])
            
            scaler.step(optimizer)
            scaler.update()
            
            train_mel_losses.append(mel_loss.item())
            train_gate_losses.append(gate_loss.item())
            train_emotion_losses.append(emotion_loss.item())
            
            # train_tqdm.set_postfix({"mel_loss":mel_loss.item(), "emotion_loss":emotion_loss.item(),"gate_loss":gate_loss.item(), "step": step})
            step += 1
                    
            if is_main_process() and step % int(config["save_checkpoint_per_steps"]) == 0:
                state_dict_path = f'{config["checkpoint_dir"]}/checkpoint_{step}.pt'
                save_checkpoint(model, optimizer, step, state_dict_path)
                
            if is_main_process() and step % int(config["logging_per_steps"]) == 0:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    val_mel_loss, val_gate_loss, val_emotion_loss = validate(model, criterion, val_loader, step)
                train_mel_loss, train_gate_loss, train_emotion_loss = \
                    statistics.mean(train_mel_losses), statistics.mean(train_gate_losses), statistics.mean(train_emotion_losses)
                    
                writer.add_scalars('loss/mel', {"train": train_mel_loss, "val": val_mel_loss}, step)
                writer.add_scalars('loss/gate', {"train": train_gate_loss, "val": val_gate_loss}, step)
                writer.add_scalars('loss/emotion', {"train": train_emotion_loss, "val": val_emotion_loss}, step)
        
        if is_main_process():
            print("###train loss at epoch={}: \nmel_loss={:4f} gate_loss={:4f} emotion_loss={:4f}\n" \
                .format(epoch, statistics.mean(train_mel_losses), statistics.mean(train_gate_losses), statistics.mean(train_emotion_losses)))
            
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
    
    config = load_yaml(config_path)
    train(config)