# -*- coding: utf-8 -*-
import tensorflow as tf
import mlp_input_data
import mlp_inference


Batch_Size = 10000
Learning_Rate_Base = 0.8
Learning_Rate_Decay = 0.96
Regularaztion_Rate = 0.0001
Moving_Average_Decay = 0.99
Train_steps = 180
Data_Path = "/home/fhq/data"
Model_Save_Path = "/home/fhq/rnn/model"
Model_Name = "Model.ckpt"
N_GPU = 2


def get_loss(x, y_, regularizer, scope):
    y = mlp_inference.inference(x, regularizer)
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    return loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
        return average_grads


def _train(_x_, _y_):
    '''
    use multi_gpu to calc loss
    use opt to optimize
    '''
    regularizer = tf.contrib.layers.l2_regularizer(Regularaztion_Rate)
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0),
        trainable= False)
    learning_rate = tf.train.exponential_decay(
        Learning_Rate_Base, global_step, 1,
        Learning_Rate_Decay)
    opt = tf.train.AdamOptimizer(learning_rate)
    tower_grads = []

    for i in range(N_GPU):
        with tf.device('gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                cur_loss = get_loss(_x_, _y_, regularizer, scope)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

    # calc ave and push to tensor_board
    grads = average_gradients(tower_grads)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(
                'gradients_on_average/%s' % var.op.name, grad)

    # update
    apply_gradient_op = opt.apply_gradients(
        grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # calc moving ave
    variable_averages = tf.train.ExponentialMovingAverage(
        Moving_Average_Decay, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variable_averages_op)

    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    log_device_placement=True)) as sess:
        init.run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_writer = tf.summary.FileWriter(Model_Save_Path,
                                               sess.graph)
        for step in range(Train_steps):
            # training
            _, loss_value = sess.run([train_op, cur_loss])
            if step % 10 == 0 or (step + 1) == Train_steps:
                result = ('step %d, loss=%.2f')
                print(result % (step, loss_value))
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary, step)
                saver.save(sess, Model_Save_Path +
                           Model_Name, step)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    _x_, _y_ = mlp_input_data.read_data(Data_Path, 'train')
    _train(_x_, _y_)


if __name__ == '__main__':
    tf.app.run()
