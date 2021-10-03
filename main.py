"""
This program implements two algorithms for converting a discrete probability
distribution into an epsilon-soft probability distribution with minimal
KL-divergence from the original.

The function `soften` does this directly and the function `tf_soften` does this
implicitly using gradient descent to provide evidence that `soften` satisfies
the criteria described above.
"""


import tensorflow as tf
import numpy as np


def soften(probs, epsilon):
    """Makes the given probability distribution epsilon-soft.

    Converts distribution P to the epsilon-soft distribution Q
    that minimizes KL-divergence(P, Q).

    Proof:
    no proof, compare results with tf_soften for empirical evidence.

    The important thing to note is that the amount that resulting
    probabilities are decreased, to make up for low probabilities being
    increased to epsilon, is spread as equally as possible.

    For example, soften([.1, .35, .55], .2) produces [.2, .3, .5].
    """
    probs = list(probs)
    diff = 1.0
    while diff > 0:
        diff = 0.0
        num_greater = 0
        for i in range(len(probs)):
            if probs[i] < epsilon:
                diff += epsilon - probs[i]
                probs[i] = epsilon
            elif probs[i] > epsilon:
                num_greater += 1
        add = diff / num_greater
        for i in range(len(probs)):
            if probs[i] > epsilon:
                probs[i] -= add
    return probs


def KL_divergence(P, Q):
    """Calculates the KL-divergence from Q to P.
    """
    return tf.reduce_sum(P * tf.math.log(P / Q), axis=-1)


def tf_soften(probs, epsilon, n=200, a=1e1, b=1e1, c=1e0, learning_rate=1e-5, momentum=0.0):
    """Creates an epsilon-soft distribution with minimal KL-divergence.

    Uses gradient descent to force the resulting distribution to be a valid
    epsilon-soft distribution while minimizing the KL-divergence from the
    original distribution.
    """
    target_total = tf.constant(1. - len(probs) * epsilon, 'float32')
    epsilon = tf.constant(epsilon, 'float32')
    orig_probs = tf.constant(probs, 'float32')
    probs = tf.Variable(orig_probs, dtype='float32')
    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum)

    def loss_fn(probs, a, b, c):
        """Returns the loss for the given probability distribution.
        a, b, and c are weights for combining the different losses
        """
        # this loss raises probabilities that are below epsilon
        error_soft = tf.reduce_mean(tf.abs(tf.maximum(0, epsilon - probs)))
        # this loss makes the probabilities add to 1
        error_sum = tf.abs(1. - tf.reduce_sum(probs, axis=-1))
        # this loss is simply the KL-divergence from the new to the original
        kld = KL_divergence(orig_probs, probs)
        # combine the losses and return
        return a * error_soft + b * error_sum + c * kld

    def improve_step():
        """Runs one step of gradient descent on the probabilities
        """
        with tf.GradientTape() as d2dx2_tape:
            with tf.GradientTape() as ddx_tape:
                loss = loss_fn(probs, a, b, c)
            ddx = ddx_tape.gradient(loss, probs)
        d2dx2 = d2dx2_tape.gradient(ddx, probs)
        history.append(loss.numpy())
        grad = ddx / tf.sqrt(tf.reduce_sum(tf.square(d2dx2)))
        optimizer.apply_gradients([(grad, probs)])

    # keep track of losses over the course of training
    history = []

    # build up enough history to consider stopping
    for i in range(n + 1):
        improve_step()

    # improve until we're not getting any better
    while np.mean(history[-n:]) < np.mean(history[-n-1:-1]):
        improve_step()

    return probs.numpy(), history


if __name__ == '__main__':
    NUM_EVENTS = 4
    VERBOSE = 1
    NUM_SAMPLES = 1000

    sample_diffs = []
    for i in range(NUM_SAMPLES):
        original_probs = np.random.random(size=NUM_EVENTS).astype('float32')
        original_probs /= original_probs.sum()
        epsilon = np.random.random() / len(original_probs)
        if VERBOSE >= 2:
            print('original probabilities:', list(original_probs))
            print('epsilon:', epsilon)
        soft_probs = np.float32(soften(original_probs, epsilon))
        if VERBOSE >= 3:
            print('epsilon-soft probabilities:', soft_probs)
        tf_soft_probs, history = tf_soften(original_probs, epsilon)
        if VERBOSE >= 3:
            print('TF epsilon-soft probabilities:', tf_soft_probs)
        sample_diffs.append(soft_probs - tf_soft_probs)
        if VERBOSE >= 2:
            print('error:', soft_probs - tf_soft_probs)
            print('mean absolute error:', np.mean(np.abs(soft_probs - tf_soft_probs)))
            print('mean squared error:', np.mean(np.square(soft_probs - tf_soft_probs)))
        if VERBOSE >= 1:
            print('total mean error:', np.mean(sample_diffs))
            print('total mean absolute error:', np.mean(np.abs(sample_diffs)))
    if VERBOSE < 1:
        print('total mean error:', np.mean(sample_diffs))
        print('total mean absolute error:', np.mean(np.abs(sample_diffs)))
