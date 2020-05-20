"""Run inference a DeepLab v3 model using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import argparse
import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt
import time
from ctypes import *


from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./dataset/img/',
                    help='The directory containing the image data.')

parser.add_argument('--transfer_data_dir', type=str, default='./dataset/transfer_data/',
                    help='The directory containing the transfer data.')

parser.add_argument('--output_dir', type=str, default='./dataset/inf/',
                    help='Path to the directory to generate the inference results')
parser.add_argument('--npy_output_dir', type=str, default='./dataset/inf_npy',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='test.txt', help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 2
class_three = ['lvpao','jiaozhi','bg']

def color_transfer(source1):
          source = cv2.imread("./dataset/16384-86016.png")
          target = cv2.cvtColor(source1, cv2.COLOR_BGR2LAB).astype("float32")
          source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")

          (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
          (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

          (l, a, b) = cv2.split(target)
          l -= lMeanTar
          a -= aMeanTar
          b -= bMeanTar

          l = (lStdSrc / lStdTar) * l
          a = (aStdSrc / aStdTar) * a
          b = (bStdSrc / bStdTar) * b

          l += lMeanSrc
          a += aMeanSrc
          b += bMeanSrc

          l = np.clip(l, 0, 255)
          a = np.clip(a, 0, 255)
          b = np.clip(b, 0, 255)


          transfer = cv2.merge([l, a, b])
          transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
          return transfer


def image_stats(image):

        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())

        return (lMean, lStd, aMean, aStd, bMean, bStd)



def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
          'class_classes': 3, # 3
      })
  
  class_model = tf.estimator.Estimator(
      model_fn=deeplab_model.pre_class_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
          'class_classes': 3,# 3
      })  
  mask_model = tf.estimator.Estimator(
      model_fn=deeplab_model.pre_mask_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
          'class_classes': 3,# 3
      })
  
  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]
  new_dir = FLAGS.transfer_data_dir
  transfer_files = []
  result_file = open('result','w')
  load_time = 0
  print(len(examples))
  for img_name in examples:
        print(img_name)
        source = cv2.imread(FLAGS.data_dir+img_name)
        transfer = color_transfer(source)
        transfer = cv2.resize(transfer,(512,512))
        path = new_dir + img_name
        transfer_files.append(path)
        cv2.imwrite(path,transfer)
        
  class_predictions = class_model.predict(
                input_fn=lambda: preprocessing.eval_input_fn(transfer_files),
                hooks=pred_hooks) 
    
  net_c1_list = []
  net_c2_list = []
  net_c3_list = []
  net_list = []
  mask_img = []
  start_time = time.time()
  for class_pred_dict,img_name in zip (class_predictions,transfer_files):
        image_basename = os.path.splitext(os.path.basename(img_name))[0]
        class_label = class_three[np.argwhere(class_pred_dict['decoded_class'] ==1)[0][0]]
        print(image_basename+'.png'+'\t'+class_label)
        result_file.write(image_basename+'.png'+'\t'+class_label+'\n')            
        result_file.flush()
        if class_label != '': #'lvpao': 
            net_c1 = class_pred_dict['pred_net_c1']
            net_c2 = class_pred_dict['pred_net_c2']
            net_c3 = class_pred_dict['pred_net_c3']
            net = class_pred_dict['pred_net']
            net_c1_npy = FLAGS.npy_output_dir +'/' +  image_basename + '_net_c1.npy'
            net_c2_npy = FLAGS.npy_output_dir +'/' +  image_basename + '_net_c2.npy'
            net_c3_npy = FLAGS.npy_output_dir +'/' +  image_basename + '_net_c3.npy'
            net_npy = FLAGS.npy_output_dir +'/' +  image_basename + '_net.npy'
            np.save(net_c1_npy,net_c1)
            np.save(net_c2_npy,net_c2)
            np.save(net_c3_npy,net_c3)
            np.save(net_npy,net)
            net_c1_list.append(net_c1_npy)
            net_c2_list.append(net_c2_npy)
            net_c3_list.append(net_c3_npy)    
            net_list.append(net_npy)
            mask_img.append(image_basename)
  load_time = load_time + time.time() - start_time
  
  mask_predictions = mask_model.predict(
                 input_fn=lambda: preprocessing.mask_input_fn(net_c1_list,net_c2_list,net_c3_list,net_list),
                hooks=pred_hooks) 
 
  start_time_1 = time.time()
  for mask_pred,img_path in zip(mask_predictions,mask_img):
        mask = mask_pred['decoded_mask']
        mask[mask==0] = 1
        mask[mask == 255] = 0
        mask[mask ==1] = 255
        mask_name = img_path + '_mask.png'
        path_mask = os.path.join(FLAGS.output_dir, mask_name)
        print(mask.shape)
        mask = Image.fromarray(mask)
        mask.save(path_mask)
        print('generate '+path_mask)
        

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = '1,2' #use GPU with ID=0  
  config = tf.ConfigProto()  
  config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM  
  config.gpu_options.allow_growth = True #allocate dynamically  
  sess = tf.Session(config = config)  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
