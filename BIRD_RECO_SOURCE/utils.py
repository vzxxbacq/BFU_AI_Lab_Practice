#!coding=utf-8
import os
import wave
import numpy
from numpy import linalg
import sidekit

class BasicUtils:

    def __init__(self):
        pass

    def read_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        return lines

    def write_file(self, filepath, content_lines):
        with open(filepath, 'w') as f:
            f.writelines(content_lines)
        print("write in %s success" % (filepath))

    def get_wav_duration(self, wav_filepath):
        f = wave.open(wav_filepath)
        sample_frequency = f.getframerate()
        num_samples = f.getnframes()
        duration = num_samples / float(sample_frequency)
        return duration

    def trans_txt2array(self, ivec_filepath):
        '''
        :param filepath the txt file path
        :return: the all ivector arrays <type: dict>
        '''
        ivector_arrays = {}
        trans_lines = self.read_file(ivec_filepath)
        for line in trans_lines:
            splits = line.strip().split(' ')
            key = splits[0]  # gender or uttId
            text_array = splits[3:len(splits) - 1]  # 400
            number_array = numpy.array([eval(s) for s in text_array])
            ivector_arrays[key] = number_array
        return ivector_arrays

    def compute_cosine(self, array1, array2):
        num = float(sum(array1 * array2))
        denom = linalg.norm(array1) * linalg.norm(array2)
        cos = num / denom
        return cos

    def get_info4mfcc(self, wavscp_path, project_dir, set_name):
        wav_list = []
        input_file_list = []
        output_feat_list = []
        output_feat_lines = []
        wav_scp_lines = self.read_file(wavscp_path)
        for line in wav_scp_lines:
            uttId, filepath = line.strip().split(' ')
            wav_list.append(uttId)
            input_file_list.append(filepath)
            output_feat_list.append(project_dir + '/mfcc/' + set_name + '/' + uttId + '.h5')
            output_feat_lines.append(uttId + ' ' + project_dir + '/mfcc/' + set_name + '/' + uttId + '.h5' + '\n')
        self.write_file(project_dir + '/data/' + set_name + '/feats.scp', output_feat_lines)

        return  wav_list, input_file_list, output_feat_list

    def make_mfcc_feats(self, wavList, input_file_list, output_feat_list, nj):

        extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                              feature_filename_structure=None,
                                              sampling_frequency=8000,
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

        print("Start extracting the features")
        channel_list = numpy.zeros(len(wavList), dtype='int')
        extractor.save_list(show_list=wavList, channel_list=channel_list, audio_file_list=input_file_list,
                            feature_file_list=output_feat_list, num_thread=nj)
        print("extracting the features success")

    def get_feature_server(self, feature_filename_structure, delta=True, double_delta=True,
                           dataset_list=["energy", "cep", "vad"], feat_norm="cmvn",
                           keep_all_features=False):

        server = sidekit.FeaturesServer(feature_filename_structure=feature_filename_structure,
                            dataset_list=dataset_list,
                            mask=None,
                            feat_norm=feat_norm,
                            delta=delta,
                            double_delta=double_delta,
                            rasta=True,
                            context=None,
                            keep_all_features=keep_all_features)
        return server
