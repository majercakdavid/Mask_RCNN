from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

x = y = np.random.randn(32, 12)  # dummy data

lrs = {}
# for lr, m_mul, t_mul in itertools.product([0.001], [1, .5], [1, 5]):
for lr, m_mul, t_mul in itertools.product([0.001], [.5], [1.5]):
    print(f'lr: {lr}, m_mul: {m_mul}, t_mul: {t_mul}')

    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step',
                              trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    items = []

    with tf.Session() as sess:
        # sess.run(tf.compat.v1.train.create_global_step())
        sess.run(tf.global_variables_initializer())
        print('GLOBAL STEP:', tf.train.get_or_create_global_step())
        # tf.compat.v1.train.get_or_create_global_step()
        # sess.run(tf.compat.v1.global_variables_initializer())

        learning_rate = tf.train.cosine_decay_restarts(learning_rate=lr,
                                                       alpha=0.001/100,
                                                       m_mul=m_mul,
                                                       t_mul=t_mul,
                                                       global_step=tf.train.get_or_create_global_step(),
                                                       first_decay_steps=10)
        ipt = Input((12,))
        out = Dense(12)(ipt)
        model = Model(ipt, out)
        model.compile(
            tf.keras.optimizers.SGD(
                momentum=0.9, learning_rate=learning_rate, clipnorm=5.0),
            loss='mse')

        for iteration in range(100):
            model.train_on_batch(x, y)

            curr_lr, curr_step = sess.run([model.optimizer._decayed_lr(
                'float32'), increment_global_step_op])
            items.append(curr_lr)
            print("lr at iteration {}: {}".format(
                iteration + 1, curr_lr))
        lrs[(lr, m_mul, t_mul)] = items

plt.figure(figsize=(16, 8))
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.title('Cosine decay learning rate with restarts')

for lr, m_mul, t_mul in lrs.keys():
    plt.plot(lrs[(lr, m_mul, t_mul)],
             label=f'm_mul: {m_mul}, t_mul: {t_mul}')

# plt.legend()
plt.show()
