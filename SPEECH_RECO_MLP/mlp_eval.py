# -*- coding: utf-8 -*-
import time
import csv
import tensorflow as tf
import mlp_input_data
import mlp_inference
import mlp_train


Eval_Interval_Time = 15
Pred_Path = "/home/fhq/rnn/data"
Result_Path = "/home/fhq/rnn/result"


def evaluate(_x_, _y_):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mlp_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mlp_inference.OUTPUT_NODE], name='y-input')
        y = mlp_inference.inference(x, None)
        validate_feed = {x: _x_, y: _y_}
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(
            mlp_train.Moving_Average_Decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                with open(Result_Path + '/result.csv', 'w') as result:
                    result_writer = csv.writer(result)
                    ckpt = tf.train.get_checkpoint_state(mlp_train.Model_Save_Path)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                        ans = "%s Steps Model, Acc is %d" % (global_step, accuracy_score)
                        result_writer.writerow(ans)
                    else:
                        print('No ckpt file found')
                        return
                time.sleep(Eval_Interval_Time)


def main(argv=None):
    _x_, _y_ = mlp_input_data.read_data(Pred_Path, 'predict')
    evaluate(_x_, _y_)


if __name__ == '__main__':
    tf.app.run()
