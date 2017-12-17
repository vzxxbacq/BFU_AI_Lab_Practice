
#!coding=utf-8
import os
import numpy
import sidekit
from utils import BasicUtils

basic_ops = BasicUtils()
nj = 10
n_feats = 63
n_components = 512
tv_rank = 400
TV_matrix_path = 'TV_matrix'

def get_idmap(wavscp_path):
    tv_idmap = sidekit.IdMap()
    models = []
    segments = []
    wavscp_lines = basic_ops.read_file(wavscp_path)
    for line in wavscp_lines:
    
        splits = line.strip().split(' ')
        uttId = splits[0]
        spkId =uttId.split('_')[0]
        models.append(spkId)
        segments.append(uttId)
        tv_idmap.leftids = numpy.asarray(models)
        tv_idmap.rightids = numpy.asarray(segments)
        tv_idmap.start = numpy.empty(tv_idmap.rightids.shape, '|O')
        tv_idmap.stop = numpy.empty(tv_idmap.rightids.shape, '|O')
        tv_idmap.validate()
    return tv_idmap
def get_stat_server(ubm, idmap, feature_server, stat_path):

    if os.path.exists(stat_path):
        print("stat server exits")
        stats = sidekit.StatServer(stat_path, distrib_nb=n_components, feature_size=n_feats)
    else:
        stats = sidekit.StatServer(idmap, distrib_nb=n_components, feature_size=n_feats)
        stats.accumulate_stat(ubm=ubm, feature_server=feature_server, seg_indices=range(stats.segset.shape[0]),num_thread=nj)
        stats.write(stat_path)
    return stats

project_dir = '/home/wcq/bird'
train_wavscp_path = os.path.join(project_dir, 'data/train/wav.scp')
enroll_wavscp_path = os.path.join(project_dir, 'data/enroll/wav.scp')
test_wavscp_path = os.path.join(project_dir, 'data/test/wav.scp')

train_feature_filename_structure = "./mfcc/train/{}.h5"
enroll_feature_filename_structure = "./mfcc/enroll/{}.h5"
test_feature_filename_structure = "./mfcc/test/{}.h5"

train_ivecs_stat_path = './exp/train_ivecs_stat'
enroll_ivecs_stat_path = './exp/enroll_ivecs_stat'
test_ivecs_stat_path = './exp/test_ivecs_stat'

train_stats_path = './task/train_stat.h5'
enroll_stats_path = './task/enroll_stat.h5'
test_stats_path = './task/test_stat.h5'

ubm_path = 'task/ubm512.h5'
ubm = sidekit.Mixture(ubm_path)

print("Acc the train stats")
train_idmap = get_idmap(train_wavscp_path)
train_feature_server = basic_ops.get_feature_server(train_feature_filename_structure)
train_stat_server = get_stat_server(ubm, train_idmap, train_feature_server, train_stats_path)

print("Train the T")
# multiprocess on one node for train T space
fa = sidekit.FactorAnalyser()
fa.total_variability(train_stats_path, ubm, tv_rank,
                     nb_iter=10, min_div=True, tv_init=None, batch_size=2000,
                     save_init=True, output_file_name=TV_matrix_path, num_thread=nj)

print("Extract train ivectors")
train_ivecs_stat = fa.extract_ivectors(ubm, train_stats_path, uncertainty=False)
train_ivecs_stat.write(train_ivecs_stat_path)

print("Extract enroll ivectors")
enroll_idmap = get_idmap(enroll_wavscp_path)
enroll_feature_server = basic_ops.get_feature_server(enroll_feature_filename_structure)
enroll_stat_server = get_stat_server(ubm, enroll_idmap, enroll_feature_server, enroll_stats_path)

enroll_ivecs_stat = fa.extract_ivectors(ubm, enroll_stats_path, uncertainty=False)
enroll_ivecs_stat.write(enroll_ivecs_stat_path)

print("Extract test ivectors")
test_idmap = get_idmap(test_wavscp_path)
test_feature_server = basic_ops.get_feature_server(test_feature_filename_structure)
test_stat_server = get_stat_server(ubm, test_idmap, test_feature_server, test_stats_path)

test_ivecs_stat = fa.extract_ivectors(ubm, test_stats_path, uncertainty=False)
test_ivecs_stat.write(test_ivecs_stat_path)

print("ivectors Done!")



