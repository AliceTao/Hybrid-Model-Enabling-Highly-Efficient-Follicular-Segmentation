"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from functools import reduce
from operator import mul
from own import resnet_own
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
#from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from own import conv_own
from utils import preprocessing
import numpy as np
import math
import tensorflow.contrib.slim as slim
np.set_printoptions(threshold=np.inf)
from PIL import Image

# loss
from own import loss_own

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4

_CROSS_LAMBDA = 0.1





def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)




def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        with tf.device('/gpu:1'):
          conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
          conv_3x3_1 = conv_own.resnet_utils_conv2d_same(inputs, depth, 3, stride=1, rate=6, scope='conv_3x3_1')#atrous_rates[0] resnet_utils.conv2d_same
          conv_3x3_2 = conv_own.resnet_utils_conv2d_same(inputs, depth, 3, stride=1, rate=12, scope='conv_3x3_2')#atrous_rates[1] resnet_utils.conv2d_same
          conv_3x3_3 = conv_own.resnet_utils_conv2d_same(inputs, depth, 3, stride=1, rate=18, scope='conv_3x3_3')#atrous_rates[2] resnet_utils.conv2d_same

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          with tf.device('/gpu:1'):
          # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
            image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def deeplab_v3_generator(num_classes,
                         class_classes,
                         output_stride,
                         base_architecture,
                         pre_trained_model,
                         batch_norm_decay,
                         data_format='channels_last'):
  if data_format is None:
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    pass

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
    raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

  if base_architecture == 'resnet_v2_50':
    base_model = resnet_v2.resnet_v2_50
  else:
    base_model = resnet_v2.resnet_v2_101

  def model_class(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        logits, end_points = base_model(inputs,
                                      num_classes=None,
                                      is_training=is_training,
                                      global_pool=False,
                                      output_stride=output_stride)
    inputs_size = tf.shape(inputs)[1:3]
    with tf.variable_scope("classification"):
        with tf.device('/gpu:1'):
            class_net = end_points[base_architecture+'/block1']
            conv5 = conv(class_net, 3, 3, 256, 4 ,4, padding='VALID',name='conv5')
            norm5 = lrn(conv5, 2, 1e-04, 0.75, name='norm5')
            pool5 = max_pool(norm5, 3, 3, 2, 2, padding='VALID', name='pool5')
            flattened = tf.reshape(pool5, [-1, 7*7*256])
            fc6 = fc(flattened, 7*7*256, 4096, name='fc6')
            dropout6 = dropout(fc6, 0.5)
            fc7 = fc(dropout6, 4096, 4096, name='fc7')
            dropout7 = dropout(fc7, 0.5)
                
            class_logits = fc(dropout7, 4096, class_classes,relu=False, name='fc8')
            
            
            
    return class_logits,end_points[base_architecture + '/block1'],end_points[base_architecture + '/block2'],end_points[base_architecture + '/block3'],end_points[base_architecture + '/block4'],inputs_size

  def model_mask(net_c1,net_c2,net_c3,net,inputs_size,is_training): 
    with tf.variable_scope("cut_mask"):
        with tf.device('/gpu:0'):
            net_c1 = net_c1#end_points[base_architecture + '/block3']
            net_c2 = net_c2
            net_c3 = net_c3
            net = net #end_points[base_architecture + '/block4']
            with tf.variable_scope("add_logits"):
                net_c = layers_lib.conv2d(net_c3, 256, [1,1], stride=1,activation_fn=None, normalizer_fn=None, scope='conv_1x1')

            net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
            with tf.variable_scope("upsampling_logits"):
                net = layers_lib.conv2d(net, 256, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1_add')
                net = tf.concat([net, net_c], axis=3, name='concat')
                net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            mask_logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')

    return mask_logits

  return model_class,model_mask




def deeplabv3_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""

  if isinstance(features, dict):
    features = features['feature']
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)

  network_class,network_mask = deeplab_v3_generator(params['num_classes'],
                                 params['class_classes'],
                                 params['output_stride'],
                                 params['base_architecture'],
                                 params['pre_trained_model'],
                                 params['batch_norm_decay'])
  
  
  gt_mask = labels['label']
  gt_class = labels['class']
  # class detection
  class_logits,net_c1,net_c2,net_c3,net,inputs_size = network_class(features, mode == tf.estimator.ModeKeys.TRAIN)
  
        
  pred_class = tf.nn.softmax(class_logits)
  pred_decoded_class = tf.py_func(preprocessing.decode_class,
                                    [pred_class, params['batch_size']],
                                    tf.uint8) 
    

#   # mask detection
  mask_logits = network_mask(net_c1,net_c2,net_c3,net,inputs_size,mode == tf.estimator.ModeKeys.TRAIN)
    
  pred_mask = tf.expand_dims(tf.argmax(mask_logits, axis=3, output_type=tf.int32), axis=3)
  pred_decoded_mask = tf.py_func(preprocessing.decode_labels,
                                   [pred_mask, params['batch_size'], params['num_classes']],
                                   tf.uint8)
  predictions = {
      'decoded_mask': pred_decoded_mask,
      'decoded_class':pred_decoded_class
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
#     del predictions_without_decoded_labels['decoded_mask']
    del predictions_without_decoded_labels['decoded_class']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })
    
    
    
  

  # mask accuracy
  gt_decoded_mask = tf.py_func(preprocessing.decode_labels,
                                 [gt_mask, params['batch_size'], params['num_classes']], tf.uint8)
  gt_mask = tf.squeeze(gt_mask, axis=3)  # reduce the channel dimension.
  logits_by_num_classes = tf.reshape(mask_logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(gt_mask, [-1, ])
  valid_indices = tf.to_int32(labels_flat<= params['num_classes']-1)#labels_flat  params['num_classes']-1
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]#labels_flat 
  preds_flat = tf.reshape(pred_mask, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])
#   with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
# #     print(sess.run(valid_indices))    
#     print(sess.run(valid_preds))


  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix


    
  if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
  else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]
#   print(train_var_list)
    
  
  mask_accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
    

  tf.identity(mask_accuracy[1], name='mask_px_accuracy')
  tf.summary.scalar('mask_px_accuracy', mask_accuracy[1])

  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result


  train_mean_iou = compute_mean_iou(mean_iou[1])
  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)


  mask_recall = tf.metrics.recall(valid_labels,valid_preds)
  tf.identity(mask_recall[1], name='mask_recall')
  tf.summary.scalar('mask_recall', mask_recall[1])
  train_recall = mask_recall[0]



  # class accuracy
  y_true_cls = tf.argmax(gt_class, dimension=1)
  y_pred_cls = tf.argmax(pred_class, dimension=1) 
  gt_decoded_class = tf.py_func(preprocessing.decode_class,
                                 [gt_class, params['batch_size']], tf.uint8)
    
  class_correct_prediction = tf.equal(y_pred_cls, y_true_cls)  
  clas_accuracy = tf.reduce_mean(tf.cast(class_correct_prediction, "float"))
  class_correct_prediction = tf.metrics.accuracy(y_pred_cls, y_true_cls)
  class_accuracy =class_correct_prediction
  tf.identity(clas_accuracy, name='class_accuracy')
  tf.summary.scalar('class_accuracy', clas_accuracy)
  tf.identity(class_accuracy[1], name='class_px_accuracy')
  tf.summary.scalar('class_px_accuracy', class_accuracy[1])
  

  metrics = {'mask_px_accuracy': mask_accuracy, 'mean_iou': mean_iou,'class_accuracy':class_accuracy,'mask_recall':mask_recall}  #'mask_px_accuracy': mask_accuracy, 'mean_iou': mean_iou,
    
    
    

  logits_flat = tf.reshape(tf.argmax(mask_logits, axis=3, output_type=tf.int32)*2,[-1,])
  flat = tf.to_int32(labels_flat<=1) + logits_flat
  ### version 1
#   labels_flat = tf.to_int32(labels_flat<=1)
  loss_indices_lvpao = tf.to_int32((flat-2)>0)#257
  loss_logits_lvpao = tf.dynamic_partition(logits_by_num_classes, loss_indices_lvpao, num_partitions=2)[1]
  loss_labels_lvpao = tf.dynamic_partition(labels_flat, loss_indices_lvpao, num_partitions=2)[1]#labels_flat 
  loss_indices_bg = tf.to_int32(flat<=0)#1
  loss_logits_bg = tf.dynamic_partition(logits_by_num_classes, loss_indices_bg, num_partitions=2)[1]
  loss_labels_bg = tf.to_int32(tf.dynamic_partition(labels_flat, loss_indices_bg, num_partitions=2)[1]>0)#labels_flat 
  loss_indices_miss = tf.to_int32((flat-2+loss_indices_bg*2)<0)#256
  loss_logits_miss = tf.dynamic_partition(logits_by_num_classes, loss_indices_miss, num_partitions=2)[1]
  loss_labels_miss = tf.dynamic_partition(labels_flat, loss_indices_miss, num_partitions=2)[1]#labels_flat
  loss_indices_more = tf.to_int32((flat+loss_indices_lvpao*2-1)>0)#2
  loss_logits_more = tf.dynamic_partition(logits_by_num_classes, loss_indices_more, num_partitions=2)[1]
  loss_labels_more = tf.to_int32(tf.dynamic_partition(labels_flat, loss_indices_more, num_partitions=2)[1]>0)#labels_flat
  loss_labels_lvpao = tf.cast(loss_labels_lvpao,tf.int32)
  loss_labels_bg = tf.cast(loss_labels_bg,tf.int32)
  loss_labels_more = tf.cast(loss_labels_more,tf.int32)  
  loss_labels_miss = tf.cast(loss_labels_miss,tf.int32)   
    
  cross_entropy_lvpao = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=loss_logits_lvpao, labels=loss_labels_lvpao))
  cross_entropy_bg = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=loss_logits_bg, labels=loss_labels_bg))
  cross_entropy_more = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=loss_logits_more, labels=loss_labels_more))
  cross_entropy_miss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=loss_logits_miss, labels=loss_labels_miss)) 
  mask_cross_entropy = (cross_entropy_lvpao+ 2*train_recall*cross_entropy_miss/(1 + train_recall)
                        + cross_entropy_bg + cross_entropy_more)/(1+2*train_mean_iou*train_recall/(train_mean_iou + train_recall))



  class_lambda =tf.matmul( tf.to_float(gt_class), tf.constant([0.5,0.9,0.9], shape = [3,1]))
  class_cross_entropy = tf.reduce_mean(slim.losses.softmax_cross_entropy(class_logits, gt_class))
  w = 0.7
  cross_entropy= w*class_cross_entropy +  (1-w)*mask_cross_entropy
  
  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.identity(mask_cross_entropy, name='mask_cross_entropy')
  tf.summary.scalar('mask_cross_entropy', mask_cross_entropy)
  tf.identity(class_cross_entropy, name='class_cross_entropy')
  tf.summary.scalar('class_cross_entropy', class_cross_entropy)
    
    
  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])

    
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    

    if params['learning_rate_policy'] == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'],
          tf.cast(global_step, tf.int32) - params['initial_global_step'],
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')
    
    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
  else:
    train_op = None
    
    
  
    
    
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)




def pre_class_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""

  if isinstance(features, dict):
    features = features['feature']
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)

  network_class,network_mask = deeplab_v3_generator(params['num_classes'],
                                 params['class_classes'],
                                 params['output_stride'],
                                 params['base_architecture'],
                                 params['pre_trained_model'],
                                 params['batch_norm_decay'])

  # class detection
  class_logits,net_c1,net_c2,net_c3,net,inputs_size = network_class(features, mode == tf.estimator.ModeKeys.TRAIN)

        
  pred_class = tf.nn.softmax(class_logits)
  pred_decoded_class = tf.py_func(preprocessing.decode_class,
                                    [pred_class, params['batch_size']],
                                    tf.uint8) 

  pred_net_c1 = tf.py_func(preprocessing.decode_net,
                                    [net_c1,256, params['batch_size']],
                                    tf.uint8) 
  pred_net_c2 = tf.py_func(preprocessing.decode_net,
                                    [net_c2,512, params['batch_size']],
                                    tf.uint8) 
  pred_net_c3 = tf.py_func(preprocessing.decode_net,
                                    [net_c3,1024, params['batch_size']],
                                    tf.uint8) 
  pred_net = tf.py_func(preprocessing.decode_net,
                                    [net,2048, params['batch_size']],
                                    tf.uint8) 

    
    
  predictions = {
      'pred_net_c1': pred_net_c1,
      'pred_net_c2': pred_net_c2,
      'pred_net_c3': pred_net_c3,
      'pred_net':pred_net,
      'decoded_class':pred_decoded_class
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_class']
    return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions
                                              ,export_outputs=                
                                              {'preds':tf.estimator.export.PredictOutput(predictions_without_decoded_labels)})
        
        
def pre_mask_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""

  network_class,network_mask = deeplab_v3_generator(params['num_classes'],
                                 params['class_classes'],
                                 params['output_stride'],
                                 params['base_architecture'],
                                 params['pre_trained_model'],
                                 params['batch_norm_decay'])
  inputs = tf.zeros([1,512,512,3])
  inputs_size = tf.shape(inputs)[1:3]
  mask_logits = network_mask(features['net_c1'],features['net_c2'],features['net_c3'],features['net'],inputs_size,mode == tf.estimator.ModeKeys.TRAIN)
  pred_mask = tf.expand_dims(tf.argmax(mask_logits, axis=3, output_type=tf.int32), axis=3)
  pred_decoded_mask = tf.py_func(preprocessing.decode_labels,
                                   [pred_mask, params['batch_size'], params['num_classes']],
                                   tf.uint8)
  predictions = {
      'decoded_mask': pred_decoded_mask
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_mask']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })
