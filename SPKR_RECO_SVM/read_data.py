import Mfcc
import os
import wave
import numpy as np


def get_input(infile, name):
    '''
    :param infile: url of data
    :param name: type of data
    :return: mel, id
    '''
    spkr_id = []
    ids = 0
    mel_file = []
    now_dir = os.path.join(infile, name)
    now_files = os.listdir(now_dir)
    for file in now_files:
        ids += 1
        if file.split('.')[-1] == 'wav':
            wav_reader = wave.open(file)
            n_frames = wav_reader.getnframes()
            frames_str = wav_reader.readframes(n_frames)
            frames_nda = np.fromstring(frames_str, dtype=np.short)
            mel_data = Mfcc.calcMFCC_delta_delta(frames_nda)
            mel_data = mel_data.tolist()
            for num in range(mel_data.shape[0]):
                spkr_id.append(ids)
            for row in mel_data:
                mel_file.append(row)
    spkr_id = np.array(spkr_id)
    spkr_id.shape = -1, 1
    mel_file = np.array(mel_file)
    mel_file.shape = -1, 39
    return mel_file, spkr_id
