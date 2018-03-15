
import tensorflow as tf
import numpy as np

__all__ = [
  'NimaUtils', 
  'nima_vgg_16',
  'slim_learning_create_train_op_with_manual_grads',
]

""" _CDF in tensorflow """


# this implementation ignores the weighted sum of rating values and uses the ratings as given
# I believe this is correct because CDF(rating) exists independent scores from weighted ratings
def _cum_CDF (x): 
  # e.g. cdf([1,1,1,1]) ==  [ 0.25,  0.5 ,  0.75,  1.  ]  
  x = tf.to_float(x)
  cs = tf.cumsum(x, axis=1, reverse=False)
  total = cs[:,-1:]           # last column == cumulative sum
  cdf = tf.divide(cs, total)
  return cdf

def _emd(y, y_hat, reduce_mean=True, r=2):
    """Returns the earth mover distance between to arrays of ratings, 
    based on cumulative distribution function
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n
      reduce_mean: apply tf.reduce_mean()
      r: r=2 for rmse loss (default) or r=1 for absolute val

    Returns:
      float 
    """
    m,n = tf.convert_to_tensor(y).get_shape().as_list()
    cdf_loss = tf.subtract(_cum_CDF(y), _cum_CDF(y_hat))
    if reduce_mean:
      emd_loss = tf.pow( tf.reduce_mean( tf.pow(cdf_loss, r) ), 1/r)
      return emd_loss
    else:
      emd_loss = tf.pow( tf.reduce_mean( tf.pow(cdf_loss, r), axis=1 ), 1/r)
      return  tf.reshape(emd_loss, [m,1])


class NimaUtils(object):
  """Helper Class for Nima calculations
    NimaUtils.emd(y, y_hat) return float
    NimaUtils.score( y ) returns [[mean, std]]
  """
  @staticmethod
  def emd(y, y_hat, reduce_mean=True, r=2):
    return _emd(y, y_hat, reduce_mean)

  @staticmethod
  def mu(y):
    """mean quality score for ratings
    
    Args:
      y: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n

    Returns:
      array of [mean] floats for each row in y
    """
    m,n = tf.convert_to_tensor(y).get_shape().as_list()  
    y = tf.to_float(y)
    cs = tf.reduce_sum(tf.cumsum(y, axis=1, reverse=True), axis=1)
    total = tf.reduce_sum(y, axis=1)
    return tf.reshape(cs/total, [m,1] ) # (None,1) [m,1]
  
  @staticmethod
  def sigma(y):
    """standard deviation of ratings
    
    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores 
                shape = (None, n), array of count of scores for score from 1..n

    Returns:
      array of [stddev] floats for each row in y
    """    
    y = tf.convert_to_tensor(y)
    m,n = y.get_shape().as_list()    
    mean = NimaUtils.mu(y)
    s = tf.range(1, n+1 , dtype=tf.float32)
    p_score = tf.divide(y, tf.reshape(tf.reduce_sum(y, axis=1),[m,1]))
    stddev = tf.sqrt(tf.reduce_sum( tf.multiply(tf.square(tf.subtract(s,mean)),p_score), axis=1))
    return tf.reshape(stddev, [m,1])

  @staticmethod
  def score(y):
    """returns [mean quality score, stddev] for each row"""
    y = tf.to_float(y)
    return tf.concat([NimaUtils.mu(y), NimaUtils.sigma(y)], axis=1)

  @staticmethod
  def rmse_score(y, y_hat):
    """returns the rmse [mean, stddev] values for a mini-batch y, y_hat"""
    score_losses = NimaUtils.score(y) - NimaUtils.score(y_hat)
    rmse_scores = tf.sqrt(tf.reduce_mean(tf.square(score_losses), axis=0))
    return tf.squeeze(rmse_scores)

  @staticmethod
  def np_mu(y):
    """from https://github.com/titu1994/neural-image-assessment/blob/master/utils/score_utils.py
    accepts ratings as float, use for softmax
    args:
      y: array of shape [?,n], dtype=np.float
    """
    si = np.arange(1,11,1, dtype=np.float32)
    N = np.sum(y, axis=1)
    mean = np.sum(y * si, axis=1)/N
    return mean

  @staticmethod
  def np_sigma(y, mean=None):
    """from https://github.com/titu1994/neural-image-assessment/blob/master/utils/score_utils.py
    accepts ratings as float, use for softmax
    args:
      y: array of shape [?,n], dtype=np.float
    """
    si = np.arange(1, 11, 1, dtype=np.float32)               # [1, n]
    if mean is None: mean = NimaUtils.np_mu(y)  # [m,]
    mean = np.reshape(mean, [-1,1])    # [m,1]
    N = np.sum(y, axis=1)
    std = np.sqrt( np.sum(((si - mean) ** 2) * y, axis=1) /N  )
    return std

  @staticmethod
  def np_score(y):
    """ numpy calculation of [mean, std] from ratings as float, accepts softmax output
    args:
      y: array of shape [?,n], dtype=np.float
    """    
    mean = NimaUtils.np_mu(y)
    std = NimaUtils.np_sigma(y, mean)
    return np.transpose([mean, std])      

  @staticmethod
  def spearman_rank(y, y_hat):
    """returns the Spearman Ranked Correlation Coefficinet of the batch
      see: https://stackoverflow.com/questions/38487410/possible-to-use-rank-correlation-as-cost-function-in-tensorflow
      https://geographyfieldwork.com/SpearmansRank.htm 

      Returns tf.float32 [-1..1]
    """
    try:
      y = tf.to_float(y)
      m,n = y.get_shape().as_list()
    except:
      m,n = np.shape(y)

    if n > 1:
      # derive stddev from ratings
      y_hat_batch = tf.squeeze(NimaUtils.mu(y_hat))
      y_batch = tf.squeeze(NimaUtils.mu(y))
    else:
      # assume we already have stddev values
      y_hat_batch = tf.squeeze(y_hat)
      y_batch = tf.squeeze(y)
    
    y_hat_rank = tf.nn.top_k(y_hat_batch, k=m, sorted=True, name='y_hat_rank').indices
    y_rank = tf.nn.top_k(y_batch, k=m, sorted=True, name='y_rank').indices
    rank_diffs = y_hat_rank - y_rank
    rank_diffs_squared_sum = tf.reduce_sum(rank_diffs * rank_diffs)
    six = tf.constant(6)
    one = tf.constant(1.0)
    numerator = tf.cast(six * rank_diffs_squared_sum, dtype=tf.float32)
    divider = tf.cast(m * m * m - m, dtype=tf.float32)
    spearman_batch = one - (numerator / divider)
    return spearman_batch

  @staticmethod
  def linear_correlation(y, y_hat):
    """returns the linear correlation coefficient
      see: https://www.mathway.com/examples/statistics/correlation-and-regression/finding-the-linear-correlation-coefficient?id=328

    Returns: tf.float32 [-1..1]  
    """
    try:
      y = tf.to_float(y)
      m,n = y.get_shape().as_list()
    except:
      m,n = np.shape(y)

    if n > 1:
      # derive stddev from ratings
      predictions_batch = NimaUtils.mu(y_hat)
      labels_batch = NimaUtils.mu(y)
    else:
      # assume we already have stddev values
      predictions_batch = y_hat
      labels_batch = y

    sum_y = tf.squeeze(tf.reduce_sum(labels_batch, axis=0))
    sum_y2 = tf.squeeze(tf.reduce_sum(tf.square(labels_batch), axis=0))
    sum_y_hat = tf.squeeze(tf.reduce_sum(predictions_batch, axis=0))
    sum_y_hat2 = tf.squeeze(tf.reduce_sum(tf.square(predictions_batch), axis=0))
    yy_hat_matmul = tf.squeeze(tf.matmul(tf.transpose(labels_batch),predictions_batch))  # [1 m] * [ m 1] = [1 1]

    numerator = (m * yy_hat_matmul) - (sum_y * sum_y_hat)
    denominator = tf.sqrt((m*sum_y2)-tf.square(sum_y)) * tf.sqrt((m*sum_y_hat2)-tf.square(sum_y_hat))

    lcc = numerator/denominator
    return lcc



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
                x = np.append(x, [k+1])
        # print(x)
        res.append(x)
    return res

  @staticmethod
  def np_mean_score(y):
    """from https://github.com/titu1994/neural-image-assessment/blob/master/utils/score_utils.py
    accepts ratings as float, use for softmax
    args:
      y: array of shape [?,n], dtype=np.float
    """
    si = np.arange(1,11,1, dtype=np.float32)
    N = np.sum(y, axis=1)
    mean = np.sum(y * si, axis=1)/N
    return mean

  @staticmethod
  def np_std_score(y):
    """from https://github.com/titu1994/neural-image-assessment/blob/master/utils/score_utils.py
    accepts ratings as float, use for softmax
    args:
      y: array of shape [?,n], dtype=np.float
    """
    si = np.arange(1, 11, 1, dtype=np.float32)               # [1, n]
    mean = TestNimaUtils.np_mean_score(y)  # [m,]
    mean = np.reshape(mean, [-1,1])    # [m,1]
    N = np.sum(y, axis=1)
    std = np.sqrt( np.sum(((si - mean) ** 2) * y, axis=1) /N  )
    return std

  @staticmethod
  def np_score(y):
    """ numpy calculation of [mean, std] from ratings as float, accepts softmax output
    args:
      y: array of shape [?,n], dtype=np.float
    """    
    mean = TestNimaUtils.np_mean_score(y)
    std = TestNimaUtils.np_std_score(y)
    return np.transpose([mean, std])    


  @staticmethod
  def np_score_int(y):
    """ numpy calculation of [mean, std] from expanded ratings 
    accepts ratings as int only
    args:
      y: array of shape [?,n], dtype=np.int
    """
    m,n = np.shape(y)
    res = []
    x = TestNimaUtils.np_expand_ratings(y)
    for j in range(m):
        res.append([np.mean(x[j]), np.std(x[j])])
    return np.asarray(res)

  @staticmethod
  def test_score(y):
    """verify tf mu, sigma calculations match numpy"""
    tfscore = tf.Session().run( NimaUtils.score(y) )
    np.testing.assert_allclose( tfscore, TestNimaUtils.np_score_int(y) )
    print("OK NimaUtils.score() matches manual numpy calculation")

    np.testing.assert_allclose( TestNimaUtils.np_score_int(y), [TestNimaUtils.np_mean_score(y), TestNimaUtils.np_std_score(y)])
    print("OK manual np_score_int() matches np_mean_score/np_std_score calculation")
      
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



# Nima Model based on Vgg16 for training against AVA dataset
from models.research.slim.nets import vgg
from tensorflow.contrib import slim

# # copied from nets.vgg.vgg_16 with slight modifications
# def nima_vgg_16(inputs,
#            num_classes=10,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            dropout7_keep_prob=0.5,        # added kvarg to change value for dropout7 only
#            weight_decay=0.0005,
#            spatial_squeeze=True,
#            scope='vgg_16',
#            fc_conv_padding='VALID',
#            global_pool=False):
#   """Oxford Net VGG 16-Layers version D Example.

#   Note: All the fully_connected layers have been transformed to conv2d layers.
#         To use in classification mode, resize input to 224x224.

#   Args:
#     inputs: a tensor of size [batch_size, height, width, channels].
#     num_classes: number of predicted classes. If 0 or None, the logits layer is
#       omitted and the input features to the logits layer are returned instead.
#     is_training: whether or not the model is being trained.
#     dropout_keep_prob: the probability that activations are kept in the dropout
#       layers during training.
#     dropout7_keep_prob: ADDED to allow specification of value different from prior 
#       layers
#     spatial_squeeze: whether or not should squeeze the spatial dimensions of the
#       outputs. Useful to remove unnecessary dimensions for classification.
#     scope: Optional scope for the variables.
#     fc_conv_padding: the type of padding to use for the fully connected layer
#       that is implemented as a convolutional layer. Use 'SAME' padding if you
#       are applying the network in a fully convolutional manner and want to
#       get a prediction map downsampled by a factor of 32 as an output.
#       Otherwise, the output prediction map will be (input / 32) - 6 in case of
#       'VALID' padding.
#     global_pool: Optional boolean flag. If True, the input to the classification
#       layer is avgpooled to size 1x1, for any input size. (This is not part
#       of the original VGG architecture.)

#   Returns:
#     net: the output of the logits layer (if num_classes is a non-zero integer),
#       or the input to the logits layer (if num_classes is 0 or None).
#     end_points: a dict of tensors with intermediate activations.
#   """
#   # with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0005)):
#   with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
#     end_points_collection = sc.original_name_scope + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
#                         outputs_collections=end_points_collection):
#       net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#       net = slim.max_pool2d(net, [2, 2], scope='pool1')
#       net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#       net = slim.max_pool2d(net, [2, 2], scope='pool2')
#       net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#       net = slim.max_pool2d(net, [2, 2], scope='pool3')
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#       net = slim.max_pool2d(net, [2, 2], scope='pool4')
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#       net = slim.max_pool2d(net, [2, 2], scope='pool5')

#       # Use conv2d instead of fully_connected layers.
#       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                         scope='dropout6')
#       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#       # Convert end_points_collection into a end_point dict.
#       end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#       if global_pool:
#         net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
#         end_points['global_pool'] = net
#       if num_classes:
#         net = slim.dropout(net, dropout7_keep_prob, is_training=is_training,
#                           scope='dropout7')
#         net = slim.conv2d(net, num_classes, [1, 1],
#                           activation_fn=None,
#                           normalizer_fn=None,
#                           scope='fc8')
#         if spatial_squeeze and num_classes is not None:
#           net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#         end_points[sc.name + '/fc8'] = net
#       return net, end_points

# def scoped_nima_vgg_16(inputs, weight_decay=0.0005,  **kwargs):
#   with slim.arg_scope(vgg.vgg_arg_scope(weight_decay)):
#     return nima_vgg_16(inputs, **kwargs)


def slim_learning_create_train_op_with_manual_grads( total_loss, 
            optimizers,       # list of optimizers 
            grads_and_vars,   # list of grads_and_vars from optimizer.compute_gradients()
            global_step=0,                                                            
#                     update_ops=None,
#                     variables_to_train=None,
            clip_gradient_norm=0,
            summarize_gradients=False,
            gate_gradients=1,               # tf.python.training.optimizer.Optimizer.GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=False,
            gradient_multipliers=None,
            check_numerics=True):
  """Runs the training loop
      
    modified from slim.learning.create_train_op() to work with
    a matched list of optimizers and grads_and_vars

    see:
      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py
      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/training.py
  
  Returns:
      train_ops - the value of the loss function after training.
  """
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import control_flow_ops
  from tensorflow.python.training import training_util
  
  def _transform_grads_fn(grads):
      if gradient_multipliers:
          with ops.name_scope('multiply_grads'):
              grads = multiply_gradients(grads, gradient_multipliers)

      # Clip gradients.
      if clip_gradient_norm > 0:
          with ops.name_scope('clip_grads'):
              grads = clip_gradient_norms(grads, clip_gradient_norm)
      return grads
  
  if global_step is None:
      global_step = training_util.get_or_create_global_step()
      
  # we are assuming these are a matched set, should be zipped as a tuple(opt, grads, vars) 
  assert len(optimizers)==len(grads_and_vars)
  
  ### order of processing:
  # 0. grads = opt.compute_gradients() 
  # 1. grads = _transform_grads_fn(grads)
  # 2. add_gradients_summaries(grads)
  # 3. grads = opt.apply_gradients(grads, global_step=global_step) 
  
  grad_updates = []
  for i in range(len(optimizers)):
      grads = grads_and_vars[i]                               # 0. kvarg, from opt.compute_gradients()
      grads = _transform_grads_fn(grads)                       # 1. _transform_grads_fn()
      if summarize_gradients:
          with ops.name_scope('summarize_grads'):
              slim.learning.add_gradients_summaries(grads)    # 2. add_gradients_summaries()
      if i==0:
          grad_update = optimizers[i].apply_gradients( grads, # 3. optimizer.apply_gradients()
                      global_step=global_step)                #    update global_step only once
      else:
          grad_update = optimizers[i].apply_gradients( grads )
      grad_updates.append(grad_update)

  with ops.name_scope('train_op'):
      total_loss = array_ops.check_numerics(total_loss,
                                      'LossTensor is inf or nan')
      train_op = control_flow_ops.with_dependencies(grad_updates, total_loss)
      
  # Add the operation used for training to the 'train_op' collection    
  train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
      train_ops.append(train_op)
      
  return train_op

def get_step_from_latest_checkpoint(dir):
  """get global_step from checkpoint_path when outside of graph"""
  import re
  import tensorflow as tf
  path = tf.train.latest_checkpoint(dir)
  if not path:
      return 0
  found = re.findall("(\d+)$", path)
  print(found)
  print(type(found))
  return int(found[0]) if found else None    