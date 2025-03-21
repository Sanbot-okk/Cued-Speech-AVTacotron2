import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, reduced_mel_loss, reduced_lip_loss, reduced_hand_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("training.mel.loss", reduced_mel_loss, iteration)
            self.add_scalar("training.lip.loss", reduced_lip_loss, iteration)
            self.add_scalar("training.hand.loss", reduced_hand_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, reduced_mel_loss, reduced_lip_loss, reduced_hand_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_scalar("validation.mel.loss", reduced_mel_loss, iteration)
        self.add_scalar("validation.lip.loss", reduced_lip_loss, iteration)
        self.add_scalar("validation.hand.loss", reduced_hand_loss, iteration)
        _, mel_outputs, au_outputs, gate_outputs, alignments = y_pred
        mel_targets, au_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        if au_targets is not None:
            self.add_image(
                "au_target",
                plot_spectrogram_to_numpy(au_targets[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                "au_predicted",
                plot_spectrogram_to_numpy(au_outputs[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
        
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
