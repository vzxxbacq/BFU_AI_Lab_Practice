
#!coding=utf-8
import sidekit
from utils import BasicUtils
basic_ops = BasicUtils()

enroll_ivecs_stat = sidekit.StatServer("./exp/enroll_ivecs_stat", distrib_nb=512, feature_size=63)
test_ivecs_stat = sidekit.StatServer("./exp/test_ivecs_stat", distrib_nb=512, feature_size=63)

sts_per_model = enroll_ivecs_stat.mean_stat_per_model()
spk_list = sts_per_model.modelset

mean_ivecs = sts_per_model.stat1
test_ivecs = test_ivecs_stat.stat1

print(spk_list.shape, mean_ivecs.shape, test_ivecs.shape)

test_utts = test_ivecs_stat.segset
print(test_utts.shape)

result_lines = []
for k in range(len(test_utts)):
    uttId = test_utts[k]
    uttId_ivec = test_ivecs[k]
    temp_scores = []
    for i in range(len(spk_list)):
        cos = basic_ops.compute_cosine(uttId_ivec, mean_ivecs[i])
        temp_scores.append(cos)
    max_score = max(temp_scores)
    max_score_index = temp_scores.index(max_score)
    result_spk = spk_list[max_score_index]
    real_spk = uttId.split('_')[0]
    print("Cur utt is ", uttId + ' ' + real_spk + ' ' + result_spk)
    result_lines.append(uttId + ' ' + real_spk + ' ' + result_spk + '\n')

all_num = len(result_lines)
acc_num = 0.0
for line in result_lines:
    uttId, real_spk, result_spk = line.strip().split(' ')
    if real_spk == result_spk:
        acc_num += 1

print("Total Acc", float(acc_num) / all_num)
basic_ops.write_file("exp/results/ivec_gmm512_8k_cos.txt", result_lines)
