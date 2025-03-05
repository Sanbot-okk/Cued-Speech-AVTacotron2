from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, au_target, gate_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, au_out, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        if au_target is not None:
            au_target.requires_grad = False
            #au_loss = nn.MSELoss()(au_out, au_target)
            lip_loss = nn.MSELoss()(au_out[:, :10], au_target[:, :10])
            hand_loss = nn.MSELoss()(au_out[:, 10:], au_target[:, 10:])

            return mel_loss + gate_loss + lip_loss + hand_loss, mel_loss, lip_loss, hand_loss
        else:
            return mel_loss + gate_loss
