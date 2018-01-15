

import tensorflow as tf
import numpy as np

""" _CDF in tensorflow """
#
# private methods used by class NimaUtils()
#
def _weighted_score(x):
  m,n = tf.convert_to_tensor(x).get_shape().as_list()
  return tf.multiply(x, tf.range(1, n+1 , dtype=tf.float32))  # (None,10)

def _CDF (k, x):
  # assert k <= tf.shape(x)[1]
  m,n = tf.convert_to_tensor(x).get_shape().as_list()
  w_score = _weighted_score(x)        # (None,10)
  cum_k_score = tf.reduce_sum(w_score[:,:k], axis=1)  # (None)
  total = tf.reduce_sum(w_score, axis=1)  # (None)
  cdf = tf.divide(cum_k_score, total)     # (None)
  return tf.reshape(cdf, [m,1] ) # (None,1)

def _cum_CDF (x):
  # y = tf.concat( [   _CDF(i,x)    for i in tf.range(1, tf.shape(x)[1]+1) ] )
  x = tf.to_float(x)
  m,n = tf.convert_to_tensor(x).get_shape().as_list()
  y = tf.concat( [_CDF(1,x),_CDF(2,x),_CDF(3,x),_CDF(4,x),_CDF(5,x),
      _CDF(6,x),_CDF(7,x),_CDF(8,x),_CDF(9,x),_CDF(10,x)], 
      axis=1 )
  return tf.reshape(y, [m,n] )


class NimaUtils(object):
  """Help Class for Nima calculations
    NimaUtils.emd(y, y_hat) return float
    NimaUtils.score( y ) returns [[mean, std]]
  """
  @staticmethod
  def emd(y, y_hat):
    """Returns the earth mover distance between to arrays of ratings, 
    based on cumulative distribution function
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n

    Returns:
      float 
    """
    r = 2.
    m,n = tf.convert_to_tensor(y).get_shape().as_list()
    N = tf.to_float(n)
    cdf_loss = tf.subtract(_cum_CDF(y), _cum_CDF(y_hat))
    emd_loss = tf.pow( tf.divide( tf.reduce_sum( tf.pow(cdf_loss, r), axis=1 ), N), 1/r)
  #   return tf.reshape(emd_loss, [m,1])
    return tf.reduce_mean(emd_loss)

  @staticmethod
  def mu(y, shape=None):
    """mean quality score for ratings
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n

    Returns:
      array of [mean] floats for each row in y
    """
    y = tf.convert_to_tensor(y)
    m,n = y.get_shape().as_list()
    mean = tf.reduce_sum(_weighted_score(y), axis=1)/tf.reduce_sum(y, axis=1)
    return tf.reshape(mean, [m,1])
  
  @staticmethod
  def sigma(y, shape=None):
    """standard deviation of ratings
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n

    Returns:
      array of [stddev] floats for each row in y
    """    
    y = tf.convert_to_tensor(y)
    m,n = y.get_shape().as_list()    
    mean = mu(y)
    s = tf.range(1, n+1 , dtype=tf.float32)
    p_score = tf.divide(y, tf.reshape(tf.reduce_sum(y, axis=1),[m,1]))
    stddev = tf.sqrt(tf.reduce_sum( tf.multiply(tf.square(tf.subtract(s,mean)),p_score), axis=1))
    return tf.reshape(stddev, [m,1])

  @staticmethod
  def score(y):
    """returns [mean quality score, stddev] for each row"""
    return tf.concat([mu(y), sigma(y)], axis=1)


class TestNimaUtils(object):
  """
    TestNimaUtils.test_score(y)
    TestNimaUtils.test_emd_loss()
  """
  @staticmethod
  def np_expand_ratings(y):
    """manually expand nima ratings into individual ratings"""
    m,n = np.shape(y)
    res = []
    for j in range(m):
        x = []
        for k in range(n):
            for count in range(np.int(y[j,k])):
                x.append(k+1)
        res.append(x)
    return res
  
  @staticmethod
  def npscore(y):
    """ numpy calculation of [mean, std] from expanded ratings """
    m,n = np.shape(y)
    res = []
    x = TestNimaUtils.np_expand_ratings(y)
    for j in range(m):
        res.append([np.mean(x[j]), np.std(x[j])])
    return res

  @staticmethod
  def test_score(y):
    """verify tf mu, sigma calculations match numpy"""
    tfscore = tf.Session().run( NimaUtils.score(y) )
    np.testing.assert_allclose( tfscore, TestNimaUtils.npscore(y) )
    print("OK NimaUtils.score() matches manual numpy calculation")
      
  @staticmethod
  def test_emd_loss():
    a = np.ones([10], dtype=np.float32)
    b = np.arange(1,11, dtype=np.float32)
    y = np.stack([a,b])   # a,b must have the same shape
    y = np.append( y, [2*a], axis=0)
    y = np.concatenate( [y, [2*b] ] , axis=0)
    # modify y to generate y_hat
    y_hat= np.copy(y)
    y_hat[1] = y_hat[1]+2
    y_hat[3] = np.flip(y_hat[3], axis=0)
    
    sess = tf.Session()
    emd_loss = sess.run( NimaUtils.emd(y,y_hat) )
    np.testing.assert_allclose( emd_loss , 0.07387695461511612 )
    print("Confirming NimaUtils.emd_loss calculations have not changed.")
    print("OK y=%s, \ny_hat=%s, \nEMD=%s  ==0.07387695461511612" % (y, y_hat, emd_loss))


