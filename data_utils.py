import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
        self.prefix_audio = hparams.prefix_audio
        self.suffix_audio = hparams.suffix_audio
        self.prefix_visual = hparams.prefix_visual
        self.suffix_visual = hparams.suffix_visual

        self.load_visual_features = hparams.load_visual_features
        self.n_au_channels = hparams.n_au_channels

        self.mel_sampling_rate = hparams.sampling_rate / hparams.hop_length
        self.au_sampling_rate = hparams.au_sampling_rate

        random.seed(hparams.seed)
        # random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)
    
    def get_mel_text_au_triplet(self, filename_and_text):
        # separate filename and text
        filename, text = filename_and_text[0], filename_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(filename)
        au = self.get_au(filename)

        # Linear Interpolation so that the mel and AU have the same length (common decoder)
        au_interpolated = torch.nn.functional.interpolate(au.unsqueeze(0), size=(mel.shape[1]), mode='linear', align_corners=False).squeeze(0)

        return (text, mel, au_interpolated)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch("{}{}{}".format(self.prefix_audio, filename, self.suffix_audio))
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.transpose(torch.from_numpy(np.load("{}{}{}".format(self.prefix_audio, filename, self.suffix_audio))), 0, 1)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec
    
    def get_au(self, filename):
        au = torch.transpose(torch.from_numpy(np.load("{}{}{}".format(self.prefix_visual, filename, self.suffix_visual))), 0, 1)
        assert au.size(0) == self.n_au_channels, (
            'AU dimension mismatch: given {}, expected {}'.format(
                au.size(0), self.n_au_channels))

        return au

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        if self.load_visual_features:
            return self.get_mel_text_au_triplet(self.audiopaths_and_text[index])
        else:
            return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        batch: [text_normalized, mel_normalized, au] if self.load_visual_features == True
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        if len(batch[0]) > 2:
            au_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_target_len) # same length as mel
            au_padded.zero_()
        else:
            au_padded = None

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            
            if len(batch[0]) > 2:
                au = batch[ids_sorted_decreasing[i]][2]
                au_padded[i, :, :au.size(1)] = au

        return text_padded, input_lengths, mel_padded, au_padded, gate_padded, \
            output_lengths
