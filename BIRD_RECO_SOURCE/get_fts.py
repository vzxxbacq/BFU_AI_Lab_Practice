import os
import numpy as np
import sidekit

wavs_dir = '/home/wcq/bird/wavs/train'
data_dir = '/home/wcq/bird/data'
feat_fir = '/home/wcq/bird/mfcc'

test_wav_scp = os.path.join(data_dir, 'train', 'wav.scp')
test_feats_scp_paath = os.path.join(data_dir, 'train', 'feats.scp')

enroll_list = []
input_file_list = []
output_feat_list = []
output_feat_lines = []

with open(test_wav_scp, 'r') as f:
    test_wavscp_lines = f.readlines()

for line in test_wavscp_lines:
    uttId, filepath = line.strip().split(' ')
    enroll_list.append(uttId)
    input_file_list.append(filepath)
    output_feat_list.append(os.path.join(feat_fir, 'train', uttId + '.h5'))
    output_feat_lines.append(os.path.join(feat_fir, 'train', uttId + '.h5') + '\n')

with open(test_feats_scp_paath, 'w') as f:
    f.writelines(output_feat_lines)

extractor_eval = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=44100,
                                          lower_frequency=20,
                                          higher_frequency=3700,
                                          filter_bank='log',
                                          filter_bank_size=24,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad='energy',
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["energy", "cep", "vad"],
                                          keep_all_features=False
                                          )
print("Start extracting enroll features")
channel_list = np.zeros(len(enroll_list), dtype='int')
extractor_eval.save_list(show_list=enroll_list, channel_list=channel_list, audio_file_list=input_file_list,
                        feature_file_list=output_feat_list, num_thread=4)
