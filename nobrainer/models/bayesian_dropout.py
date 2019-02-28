import tensorflow as tf
import math

def bernoulli_dropout(incoming, keep_prob, mc, scale_during_training = True, name=None):
    """ Bernoulli Dropout.
    Outputs the input element multiplied by a random variable sampled from a Bernoulli distribution with either mean keep_prob (scale_during_training False) or mean 1 (scale_during_training True)
    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        scale_during_training : A boolean value determining whether scaling is performed during training or testing
        mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
        name : A name for this layer (optional).
    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """
    
    with tf.name_scope(name) as scope:

        inference = incoming

        def apply_bernoulli_dropout():
            if scale_during_training:
                return tf.nn.dropout(inference, keep_prob)
            else:
                return tf.scalar_mul(keep_prob,tf.nn.dropout(inference, keep_prob))
        
        if scale_during_training:
            expectation =  inference
        else:
            expectation =  tf.scalar_mul(keep_prob,inference)
        inference = tf.cond(mc, apply_bernoulli_dropout, lambda: expectation)
    return inference

def concrete_dropout(incoming, mc, n_filters = None, temperature=0.02, epsilon=1e-7, name='concrete_dropout'):
    """ Concrete Dropout.
    Outputs the input element multiplied by a random variable sampled from a concrete distribution
    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
        name : A name for this layer (optional).
    References:
        Concrete Dropout.
       Y. Gal, J. Hron & A. Kendall,
       Advances in Neural Information Processing Systems.
       2017.
    Links:
      [http://papers.nips.cc/paper/6949-concrete-dropout.pdf]
        (http://papers.nips.cc/paper/6949-concrete-dropout.pdf)
    """

    with tf.name_scope(name) as scope:
        inference = incoming
        print(incoming.shape)
        if n_filters != None:
            p = tf.Variable(tf.constant(0.9, shape=[n_filters],dtype=incoming.dtype),name='p')
        else:
            p = tf.Variable(tf.constant(0.9, shape=(),dtype=incoming.dtype),name='p')
        tf.add_to_collection('ps',p)
        tf.summary.histogram(p.name,p)
        p = tf.clip_by_value(
    p,
    0.05,
    0.95,
)
        def apply_concrete_dropout():
            noise = tf.random_uniform(tf.shape(inference), minval=0, maxval=1, dtype=incoming.dtype)
            z = tf.nn.sigmoid((tf.log(p+epsilon) - tf.log(1.0-p+epsilon) + tf.log(noise+epsilon) - tf.log(1.0-noise+epsilon))/temperature)
            return tf.multiply(incoming,z)
        if n_filters != None:
            expectation =  tf.multiply(p,inference)
        else:
            expectation =  tf.scalar_mul(p,inference)
        inference = tf.cond(mc, apply_concrete_dropout, lambda: expectation)
    return inference

def gaussian_dropout(incoming, keep_prob, mc, scale_during_training = True, name=None):
    """ Gaussian Dropout.
    Outputs the input element multiplied by a random variable sampled from a Gaussian distribution with mean 1 and either variance keep_prob*(1-keep_prob) (scale_during_training False) or (1-keep_prob)/keep_prob (scale_during_training True)
    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element is kept by Bernoulli dropout which is used to set the variance of the Gaussian distribution.
        scale_during_training : A boolean determining whether to match the variance of the Gaussian distribution to Bernoulli dropout with scaling during testing (False) or training (True) 
        mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
        name : A name for this layer (optional).
    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """

    with tf.name_scope(name) as scope:

        inference = incoming

        if scale_during_training:
            stddev = math.sqrt((1-keep_prob)/keep_prob)
        else:
            stddev = math.sqrt((1-keep_prob)*keep_prob)

        def apply_gaussian_dropout():
            return tf.multiply(inference,tf.random_normal(tf.shape(inference), mean = 1, stddev = stddev))
        
        inference = tf.cond(mc, apply_gaussian_dropout, lambda: inference)

    return inference

print('Bayesian dropout functions have been loaded.')