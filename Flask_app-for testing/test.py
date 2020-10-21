from __future__ import division
from flask import Flask, redirect, url_for, request 
from flask import render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import scipy.io
import glob
from PIL import Image

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print("path",APP_ROOT)

print("tensorflow version :: ",tf.__version__)

@app.route('/')
def index():
  return render_template("uploader.html")


@app.route('/upload', methods = ['POST','GET'])
def upload():
  print("upload::::::::::::::")
  inputs = os.path.join(APP_ROOT,'static/inputs/')
  preprocessed= os.path.join(APP_ROOT,'static/preprocessed/')
  outputs = os.path.join(APP_ROOT,'static/outputs/')
  reference = os.path.join(APP_ROOT,'static/reference/')
  scale = os.path.join(APP_ROOT,'static/scale/')
  gt_dir = os.path.join(APP_ROOT,'long/')
  checkpoint_dir = os.path.join(APP_ROOT,'checkpoint/')
  
  #print(target)
  if not os.path.isdir(inputs):
    os.mkdir(inputs)
  
  if not os.path.isdir(preprocessed):
    os.mkdir(preprocessed)

  if not os.path.isdir(outputs):
    os.mkdir(outputs)

  if not os.path.isdir(reference):
    os.mkdir(reference)

  if not os.path.isdir(scale):
    os.mkdir(scale)

   
  #print("requesssssssssssssssssssssssssssssttttttttttt")
  def lrelu(x):
    return tf.maximum(x * 0.2, x)


  def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


  def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


  def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

  sess = tf.Session()
  in_image = tf.placeholder(tf.float32, [None, None, None, 4])
  gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
  out_image = network(in_image)

#G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
#print("G_loss",G_loss)

  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

  print(request.files.getlist("file"))
  for upload in request.files.getlist("file"):
    print("file:::::::::::::::::",upload)
    in_path = upload
    in_fn = upload.filename
    print("in_fn::::::",in_fn)
    test_id = int(in_fn[0:5])
    print("test_id:::::::::::",test_id)
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    print("gt_fn::::::",gt_fn)

    in_exposure = float(in_fn[9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)

    
    raw = rawpy.imread(in_path)
    raw1 = raw.postprocess(no_auto_bright=True)   #short exposure input image
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio     #preprocessed input image
    pre_image=input_full
    
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)    #scaled image

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)     #reference image

    input_full = np.minimum(input_full, 1.0)
    
    output = sess.run(out_image, feed_dict={in_image: input_full})  
    output = np.minimum(np.maximum(output, 0), 1)  #output image
    loss = np.mean(np.absolute(output - gt_full))   
    
    print("loss:::::::::::::::::::",loss)
    
    output = output[0, :, :, :]
    gt_full = gt_full[0, :, :, :]
    #in_raw_full = in_raw_full[0, :, :, :]
    pre_image = pre_image[0, :, :,:]
    scale_full = scale_full[0, :, :, :]
    scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)

    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(outputs + '%5d_00_%d_out.png' % (test_id, ratio))
    scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(reference + '%5d_00_%d_out.png' % (test_id, ratio))
    scipy.misc.toimage(pre_image * 255, high=255, low=0, cmin=0, cmax=255).save(preprocessed + '%5d_00_%d_out.png' % (test_id, ratio))
    scipy.misc.toimage(raw1 , high=255, low=0, cmin=0, cmax=255).save(inputs + '%5d_00_%d_out.png' % (test_id, ratio))
    
     
  return render_template("display.html",image_name='%5d_00_%d_out.png' % (test_id, ratio),loss=loss,in_name=in_fn,gt_name=gt_fn,in_ex=in_exposure,gt_ex=gt_exposure,id=test_id)


if __name__ == '__main__':
   app.run(debug = True)
