"""NiMA neural image assessment with inception_resnet_v2 net"""
import os



# move to 'settings.py'???
### Globals
class PATH: pass



### Paths
PATH.home = os.path.join(os.getcwd())             # ./nima
PATH.slim = PATH.home + "/models/research/slim"
PATH.checkpoints = PATH.home + "/ckpt"
PATH.tmp = "/tmp"


### imports
from tensorflow.contrib import slim 
from nima_utils import slim_learning_create_train_op_with_manual_grads
os.chdir(PATH.slim)
import datasets
from nets import inception_resnet_v2 as inception
from nets import vgg
os.chdir(PATH.home)


### Helpers
# def load_pretrained_weights(net):
#   """download pretrained weights for different imageNet models
#   args:
#     net = [inception_resnet_v2 | vgg_16 | ]
#   """
#   # import settings
#   checkpoint_url={
#     'inception_resnet_v2':"http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz",
#     'vgg_16': "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
#   }
#   is_ckpt_avail = os.path.isdir(PATH.checkpoints)
#   checkpoint = checkpoint_url[net]
#   if not is_ckpt_avail:
#     print("downloading pretrained weights for {}".format(net))
#     os.makedirs(PATH.checkpoints, exist_ok=True)
#     os.chdir(PATH.tmp)
#     # download vgg_16 ckpt
#     !wget $checkpoint
#     tarfile = os.path.basename(checkpoint)
#     !tar -xvf $tarfile -C $PATH.checkpoints
#     os.remove(tarfile)
#     is_ckpt_avail = True
#   else:
#     print("{} ckpt installed".format(net))


### build model 
def net_inception(images, is_training=True, num_classes_finetune=10, finetune_dropout_keep=0.75):
  # from tensorflow.contrib import slim
  # from nets import inception_resnet_v2 as inception
  with slim.arg_scope([slim.conv2d, slim.fully_connected]):
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
      net, end_points = inception.inception_resnet_v2(images, 
                                    num_classes=None,
                                    is_training=is_training)
      end_points["baseline"] = net

      #
      # add finetune layer, with adjusted dropout_keep_prob
      #
      net = slim.flatten(net)
      net = slim.dropout(net, finetune_dropout_keep, 
                          is_training=is_training,
                          scope='Dropout')
      end_points['PreLogitsFlatten'] = net
      logits = slim.fully_connected(net, num_classes_finetune, 
                                    activation_fn=None,
                                    scope='Logits')
      end_points['finetune'] = end_points['Logits'] = logits
      end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
  return [net, end_points]

def net_vgg(images, is_training=True, num_classes_finetune=10, finetune_dropout_keep=0.75):
  # from tensorflow.contrib import slim  
  # from nets import vgg
  with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0005)):
      net, end_points = vgg.vgg_16(images, 
                          num_classes=None,
                          is_training=is_training)
      end_points["Baseline"] = net

      
      #
      # add finetune layer, with adjusted dropout_keep_prob
      #
      net = slim.flatten(net)
      net = slim.dropout(net, finetune_dropout_keep, 
                        is_training=is_training,
                        scope='dropout7')
      end_points['PreLogitsFlatten'] = net
      logits = slim.fully_connected(net, num_classes_finetune, 
                                    activation_fn=None,
                                    scope='Logits')
      end_points['finetune'] = end_points['Logits'] = logits                                    
      end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
      return [net, end_points]




### get batches
def get_batch(split="train", dataset_dir=None, file_list=None, is_training=True, batch_size=32, resized=True):
  """usage:
    images,images_raw,labels = get_batch("validation", file_list=AVA, is_training=False, batch_size=TRAIN.batch_size)

  Return:
    tf.train.batch()
  """
  # from datasets import nima, nima_ava
  dataset = datasets.nima_ava.get_split(split, 
                                dataset_dir=dataset_dir,             # dataset_dir from local fs
                                file_list=file_list,    # list() of gcloud storage urls
                                resized=resized)
  images, images_raw, labels = datasets.nima.load_batch(dataset, 
              batch_size=batch_size,
              is_training=is_training,
              resized=resized )
  return [images, images_raw, labels, dataset.num_samples]


def get_train_op(total_loss, global_step, 
                  baseline_learning_rate=3e-7,
                  finetune_learning_rate=3e-6,
                  finetune_momentum=0.9):
  """configure train_op to use separate optimizers for baseline and finetune layers
    #   see: https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow

  """
  # from nima_utils import slim_learning_create_train_op_with_manual_grads

  split_index = -2     # finetune layer weights & bias, count=2
  model = {"baseline":{}, "finetune":{}}
  # vars
  trainable = tf.trainable_variables()
  model["baseline"]["vars"] = trainable[:split_index]
  model["finetune"]["vars"] = trainable[split_index:]
  

  # grads
  gradients = tf.gradients( total_loss, trainable )
  model["baseline"]["grads"] = gradients[:split_index]
  model["finetune"]["grads"] = gradients[split_index:]
  # optimizers

  model["baseline"]["optimizer"] = tf.train.GradientDescentOptimizer(
                                  learning_rate=baseline_learning_rate)
  model["finetune"]["optimizer"] = tf.train.MomentumOptimizer(
                                  learning_rate=finetune_learning_rate,
                                  momentum=finetune_momentum
                                  )
  tf.summary.scalar("learning_rate", lr_decay["finetune"])

  grads_and_vars = [ zip(model["baseline"]["grads"], model["baseline"]["vars"]), 
                      zip(model["finetune"]["grads"], model["finetune"]["vars"]) ]
  optimizers = [ model["baseline"]["optimizer"], model["finetune"]["optimizer"] ]

  train_op = slim_learning_create_train_op_with_manual_grads(total_loss, 
                              optimizers, grads_and_vars, 
                              global_step=global_step)
  return train_op

