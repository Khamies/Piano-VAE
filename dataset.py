import os
import sys
sys.path.append('./utils/midi')

from utils.midi.midi_utils import midiread, midiwrite
from matplotlib import pyplot as plt

import torch
import numpy as np


class PianoGenerationDataset(torch.utils.data.Dataset):
    
    def __init__(self, midi_folder_path, longest_sequence_length=1491):
        
        self.midi_folder_path = midi_folder_path
        
        midi_filenames = os.listdir(midi_folder_path)
        
        self.longest_sequence_length = longest_sequence_length
        
        midi_full_filenames = map(lambda filename: os.path.join(midi_folder_path, filename), midi_filenames)
        
        self.midi_full_filenames = list(midi_full_filenames)
        
        if longest_sequence_length is None:
            
            self.update_the_max_length()
    
    def midi_filename_to_piano_roll(self, midi_filename):
        
        midi_data = midiread(midi_filename, dt=0.3)
        
        piano_roll = midi_data.piano_roll.transpose()
        
        # Pressed notes are replaced by 1
        piano_roll[piano_roll > 0] = 1
        
        return piano_roll

    def pad_piano_roll(self, piano_roll, max_length=132333, pad_value=0):
            
        original_piano_roll_length = piano_roll.shape[1]
        
        padded_piano_roll = np.zeros((88, max_length))
        padded_piano_roll[:] = pad_value
        
        padded_piano_roll[:, -original_piano_roll_length:] = piano_roll

        return padded_piano_roll
    
    def update_the_max_length(self):
        
        sequences_lengths = map(lambda filename: self.midi_filename_to_piano_roll(filename).shape[1],self.midi_full_filenames)
        
        max_length = max(sequences_lengths)
        
        self.longest_sequence_length = max_length
  
    
    def __len__(self):
        
        return len(self.midi_full_filenames)
    
    
    def __getitem__(self, index):
        
        midi_full_filename = self.midi_full_filenames[index]
        
        piano_roll = self.midi_filename_to_piano_roll(midi_full_filename)
        
        # Shifting by one time step
        sequence_length = piano_roll.shape[1] - 1
        
        # Shifting by one time step
        input_sequence = piano_roll[:, :-1]
        ground_truth_sequence = piano_roll[:, 1:]
                
        # padding sequence so that all of them have the same length
        input_sequence_padded = self.pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)
        
        ground_truth_sequence_padded = self.pad_piano_roll(ground_truth_sequence,max_length=self.longest_sequence_length,pad_value=-100)
                
        input_sequence_padded = input_sequence_padded.transpose()
        ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()
        
        return (torch.FloatTensor(input_sequence_padded),torch.LongTensor(ground_truth_sequence_padded),torch.LongTensor([sequence_length]) )