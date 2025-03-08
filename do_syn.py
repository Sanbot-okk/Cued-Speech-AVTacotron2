import os
import numpy as np
import torch
import subprocess
import random

import sys
sys.path.append('../Tacotron2/waveglow/') # used for Vocoder after synthesis

from hparams_san3 import create_hparams
from train import load_model
from text import text_to_sequence

from train import prepare_dataloaders
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# from denoiser import Denoiser

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def run_shell_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error message: {error.decode('utf-8')}")
    return output

def infer_with_tacotron2(text, model):
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, au_outputs, _, alignments = model.inference(sequence)

    return mel_outputs, mel_outputs_postnet, au_outputs, alignments

def save_outputs_model(hparams, output_directory, index_line, max_nbr_lines, mel_outputs_postnet, au_outputs, alignments,file_id):
    # Process or save the outputs as needed
    num = hparams.sampling_rate
    den = hparams.hop_length

    # Save mel_outputs_postnet
    mel_pred = mel_outputs_postnet[0]
    mel_pred_length = mel_pred.shape[1]
    mel_pred = mel_pred.cpu().data.numpy().reshape(hparams.n_mel_channels, -1).transpose()

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, file_id+"_syn.WAVEGLOW")

    # Save mel_pred with a custom binary format
    with open(output_file_path, 'wb') as fp:
        fp.write(np.asarray((mel_pred_length, hparams.n_mel_channels), dtype=np.int32))
        fp.write(mel_pred.copy(order='C'))
        fp.close()

    # Save au_outputs
    # au_pred_length = au_outputs.shape[2] # same as mel_pred_length
    au_pred = au_outputs[0]
    au_pred = au_pred.cpu().data.numpy().reshape(hparams.n_au_channels, -1).transpose()

    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, file_id+"_syn.npy")
    np.save(output_file_path, au_pred)

    """
    # Save au_pred with a custom binary format
    with open(output_file_path, 'wb') as fp:
        fp.write(np.asarray((mel_pred_length, hparams.n_au_channels, num, den), dtype=np.int32))
        fp.write(au_pred.copy(order='C'))
        fp.close()
    """
    # Save alignments
    alignments_np = alignments[0].cpu().data.numpy().transpose()

    output_file_path = os.path.join(output_directory, file_id+"_alignments.npy")
    np.save(output_file_path, alignments_np)

    print("Synthesis {}/{} OK".format(index_line, max_nbr_lines))

def main():
    random.seed(1234) # Prenet dropout is still 0.5 in eval() so still random during inference

    text_file_path = "filelists/test_cslm2023_phon.txt"  # Replace with the path to your text file
    checkpoint_path = "outdir/checkpoint_60000"
    output_directory = "outdir/inference/synthesis/"
    teacher_forcing = False # synthesis using teacher-forcing: GT frames through the prenet
    duration_forcing = False # use GT frame in the prenet every hparams.duration_forcing_periodicity frames

    hparams = create_hparams()
    hparams.sampling_rate = 22050

    # Load model
    print("Model Loading timer begins")
    tic()
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    # _ = model.cuda().eval().half() # error output format
    _ = model.cuda().eval()
    print("Model Loading timer ends")
    toc()

    num_param = get_param_num(model)
    print("Number of AVTacotron2 Parameters:", num_param)

    # Load test file
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        lines = text_file.readlines()

    if teacher_forcing:
        hparams.validation_files = text_file_path
        _, valset, collate_fn = prepare_dataloaders(hparams)
        with torch.no_grad():
            val_sampler = DistributedSampler(valset, shuffle=False) if hparams.distributed_run else None
            val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                    shuffle=False, batch_size=1,
                                    pin_memory=False, collate_fn=collate_fn)
            for i, batch in enumerate(val_loader):
                x, _ = model.parse_batch(batch)
                _, mel_outputs_postnet, au_outputs, _, alignments = model(x, duration_forcing=duration_forcing)

                # Process or save the outputs as needed
                save_outputs_model(hparams, output_directory, i+1, len(lines), mel_outputs_postnet, au_outputs, alignments)
    else:
        print('Synthesis timer begins')
        tic()
        index_line = 0
        for line in lines:
            index_line += 1
            line_parts = line.strip().split('|')
            if len(line_parts) >= 2:
                text_to_generate = line_parts[1]  # Assuming the sentence is in the second part
                _, mel_outputs_postnet, au_outputs, alignments = infer_with_tacotron2(text_to_generate, model)

            # Process or save the outputs as needed
            file_id = line_parts[0]
            save_outputs_model(hparams, output_directory, index_line, len(lines), mel_outputs_postnet, au_outputs, alignments, file_id)
        print('Synthesis timer ends')
        toc()
"""
    # Run shell commands Waveglow mel -> wav (specific to the vocoder used)
    run_shell_command(f"ls /research/crissp/sankars/AVTacotron2/{output_directory}/*.WAVEGLOW > syn_waveglow.txt")
    run_shell_command("cd /research/crissp/lengletm/Tacotron2/waveglow && "
                        f"python3 inference.py -f /research/crissp/sankars/AVTacotron2/syn_waveglow.txt -w waveglow_NEB.pt -o {output_directory} -s 0.6 --denoiser")
    run_shell_command(f"mv /research/crissp/lengletm/Tacotron2/waveglow/*.wav {output_directory}")
"""
if __name__ == "__main__":
    main()
