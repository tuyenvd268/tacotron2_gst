from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.cre_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, model_output, targets):
        emotion_targets, mel_target, gate_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        emotion_prediction, mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        
        emotion_loss = self.cre_loss(emotion_prediction, emotion_targets)

        return (mel_loss, gate_loss, emotion_loss)