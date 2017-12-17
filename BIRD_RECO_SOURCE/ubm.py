#!coding=utf-8
import os
import sidekit
import numpy
from utils import BasicUtils

basic_ops = BasicUtils()

#use python3 to run this

components_num = 512
nj = 10
n_feats = 63
project_dir = '/home/wcq/bird'

train_feature_filename_structure = "./mfcc/train/{}.h5"
enroll_feature_filename_structure = "./mfcc/enroll/{}.h5"
test_feature_filename_structure = "./mfcc/test/{}.h5"

train_wavscp_path = os.path.join(project_dir, 'data/train/wav.scp')
enroll_wavscp_path = os.path.join(project_dir, 'data/enroll/wav.scp')
test_wavscp_path = os.path.join(project_dir, 'data/test/wav.scp')

print('get train feats')
ubmList, train_input_file_list, train_output_feats_list = basic_ops.get_info4mfcc(train_wavscp_path, project_dir, 'train')
basic_ops.make_mfcc_feats(ubmList, train_input_file_list, train_output_feats_list, nj)
server_train = basic_ops.get_feature_server(train_feature_filename_structure)

print('Train the UBM by EM')
ubm = sidekit.Mixture()
llk = ubm.EM_split(server_train, ubmList, components_num, num_thread=nj, save_partial=True)
ubm.write("/home/wcq/bird/task/ubm512.h5")


ubm = sidekit.Mixture('/home/wcq/bird/task/ubm512.h5')

print('get enroll feats')
enrollList, enroll_input_file_list, enroll_output_feats_list = basic_ops.get_info4mfcc(enroll_wavscp_path, project_dir, 'enroll')
basic_ops.make_mfcc_feats(enrollList, enroll_input_file_list, enroll_output_feats_list, nj)
server_enroll = basic_ops.get_feature_server(enroll_feature_filename_structure)

#prepare the idmap for
models = []
segments = []
enroll_idmap = sidekit.IdMap()
eval_lines = basic_ops.read_file(project_dir + '/data/enroll/feats.scp')
for line in eval_lines:
    splits = line.strip().split(' ')    
    uttId = splits[0]    
    spkId = uttId.split('_')[0]
    models.append(spkId)
    segments.append(uttId)

enroll_idmap.leftids = numpy.asarray(models)
enroll_idmap.rightids = numpy.asarray(segments)
enroll_idmap.start = numpy.empty(enroll_idmap.rightids.shape, '|O')
enroll_idmap.stop = numpy.empty(enroll_idmap.rightids.shape, '|O')
enroll_idmap.validate()

print('Compute the sufficient statistics')
 # Create a StatServer for the enrollment data and compute the statistics
enroll_stat = sidekit.StatServer(enroll_idmap, components_num, n_feats)
enroll_stat.accumulate_stat(ubm=ubm, feature_server=server_enroll, seg_indices=range(enroll_stat.segset.shape[0]), num_thread=nj)

print('MAP adaptation of the speaker models')
regulation_factor = 16  # MAP regulation factor default=16
enroll_sv = enroll_stat.adapt_mean_map_multisession(ubm, regulation_factor)
enroll_sv.write('/home/wcq/bird/task/enroll_map_models.h5')

enroll_sv = sidekit.StatServer('/home/wcq/bird/task/enroll_map_models.h5', components_num, n_feats)

print('get test feats')
testList, test_input_file_list, test_output_feats_list = basic_ops.get_info4mfcc(test_wavscp_path, project_dir, 'test')
basic_ops.make_mfcc_feats(testList, test_input_file_list, test_output_feats_list, nj)
server_test = basic_ops.get_feature_server(test_feature_filename_structure)

ubm_w = ubm.w
ubm_mu = ubm.mu
ubm_cst = ubm.cst
ubm_invcov = ubm.invcov
new_stat1 = enroll_sv.stat1  # (60 * 32256)
print(new_stat1[0])
print(new_stat1[1])
spks_list = list(enroll_sv.modelset)  # (60, ), 60个spks
print("spks", len(spks_list))

def compute_likelihood_scores(testlist, feature_server):
    '''
    :param testlist: test 的 uttId 列表
    :param feature_server:  提取test feature server
    :param ubm: ubm 模型,由于map只更新了均值, 所以方差还是用ubm的, 至于权重是用 全是1 还是 ubm 的权重
    :param enroll_stat: map之后的模型,只更新了均值
    :return: likelihood score
    '''
    #print("mu shape, ", ubm.mu.shape)
    result_lines = []
    for utt in testlist:
        cep = feature_server.load(utt)[0]  #(frames, feat_dim=63)
        temp_scores = []
        for i in range(len(spks_list)):
            # for MAP, Compute the data independent term
            flatten_mu = new_stat1[i]  # print "the new mu value is ", flatten_mu
            A = (numpy.square(flatten_mu.reshape(ubm_mu.shape)) * ubm_invcov).sum(1) \
                - 2.0 * (numpy.log(ubm_w) + numpy.log(ubm_cst))
            # Compute the data independent term
            B = numpy.dot(numpy.square(cep), ubm_invcov.T) \
                - 2.0 * numpy.dot(cep, numpy.transpose(flatten_mu.reshape(ubm_mu.shape) * ubm_invcov))
            # Compute the exponential term
            lp = -0.5 * (B + A)
            pp_max = numpy.max(lp, axis=1)
            log_lk = pp_max + numpy.log(numpy.sum(numpy.exp((lp.transpose() - pp_max).transpose()), axis=1))
            temp_score = log_lk.sum()
            temp_scores.append(temp_score)
        max_score = max(temp_scores)
        max_score_index = temp_scores.index(max_score)
        result_spk = spks_list[max_score_index]
        s = utt + ' ' + str(max_score) + ' ' + str(result_spk)
        print("cur ", s)
        result_lines.append(s + '\n')
    return result_lines

result_lines = compute_likelihood_scores(testList, server_test)

all_num = len(result_lines)
acc_num = 0.0
for line in result_lines:
    splits = line.strip().split(' ')
    real_spk = splits[0].split('_')[0]
    result_spk = splits[2]
    if real_spk == result_spk:
        acc_num += 1
    else:
        print("Error: utt is ", splits[0], "result spk is: ", result_spk)
print("Total Acc", float(acc_num)/all_num)
basic_ops.write_file("/home/wcq/bird/task/all_test8k_result_infos", result_lines)
                                                           
