import tensorflow as tf


def build_network(n_features, n_actions, lr):
    """ Build the network for RL algorithm.

    :param n_features: input layer's features
    :param n_actions:  How many actions to take in output.
    :param lr: learning rate
    :return:
        [eval_net_input, target_net_input, q_target]: input
        [q_eval_net_out, loss, _train_op, q_target_net_out]: output
        [e_params, t_params]: weights
    """
    # ------------------ all inputs --------------------------
    # input for target net
    eval_net_input = tf.placeholder(tf.float32, [None, n_features], name='eval_net_input')
    # input for eval net
    target_net_input = tf.placeholder(tf.float32, [None, n_features], name='target_net_input')
    # q_target for loss
    q_target = tf.placeholder(tf.float32, [None, n_actions], name='q_target')
    # initializer
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

    # ------------------ build evaluate_net ------------------
    with tf.variable_scope('eval_net'):
        e1 = tf.layers.dense(eval_net_input, 10, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='e1')
        q_eval_net_out = tf.layers.dense(e1, n_actions, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='q_e')

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_net_out, name='TD_error'))

    with tf.variable_scope('train'):
        _train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # ------------------ build target_net --------------------
    with tf.variable_scope('target_net'):
        t1 = tf.layers.dense(target_net_input, 10, tf.nn.relu, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer, name='t1')
        q_target_net_out = tf.layers.dense(t1, n_actions, kernel_initializer=w_initializer,
                                           bias_initializer=b_initializer, name='q_t')

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    return [eval_net_input, target_net_input, q_target], [q_eval_net_out, loss, _train_op, q_target_net_out], \
           [e_params, t_params]
