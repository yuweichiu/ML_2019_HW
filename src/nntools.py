# -*- coding: utf-8 -*-
"""
Created on 2018/9/5 下午 04:10
@author: Ivan Chiu

"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope


def batch_index(b_size, total):
    lid = []
    for i in range(total//b_size):
        lid.append(i*b_size)
        # print(i, i*b_size)
    if (i + 1)*b_size < total:
        i = i + 1
        lid.append(i*b_size)
    else:
        pass
    return lid


@add_arg_scope
def input_data(x_placeholder, shape):
    """
    :param
        x_placeholder : the defined placeholder for input data.
        shape : The input tensor shape : [batch, in_height, in_width, in_channels]
                batch : the amount of input data. '-1' get non-specify.
                in_height : the height of input image.
                in_width : the width of input image.
                in_channels : the amount of channels in input image.
    """
    return tf.reshape(x_placeholder, shape=shape)

@add_arg_scope
def conv2d(name, x, depth, kernel_size, stride=1, padding='SAME', is_training=True, weight_init='he_normal', b=0.01, activation=None, norm=True):
    input_size = str(x.get_shape().as_list())
    with tf.variable_scope(name):
        in_depth = x.get_shape().as_list()[-1]
        if weight_init == 'he_normal':
            w_init = tf.initializers.he_normal()
        elif weight_init == 'glorot_uniform':
            w_init = tf.initializers.glorot_uniform()
        elif weight_init == 'glorot_normal':
            w_init = tf.initializers.glorot_normal()
        else:
            w_init = tf.initializers.glorot_normal()
        weight = tf.get_variable(name='weight', shape=[kernel_size[0], kernel_size[1], in_depth, depth], dtype=tf.float32,
                                 initializer=w_init)
        bias = tf.Variable(np.ones(shape=depth)*b, dtype=tf.float32, name='bias')
        conv = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding=padding)
        if norm is True:
            b_norm = tf.layers.batch_normalization(
                tf.nn.bias_add(conv, bias),
                axis=-1,
                training=is_training,
                trainable=True,
                momentum=0.99,
                renorm_momentum=0.9
            )
        else:
            b_norm = tf.nn.bias_add(conv, bias)

        if activation == 'relu':
            output = tf.nn.relu(b_norm)
        elif activation == 'leaky_relu':
            output = tf.nn.leaky_relu(b_norm)
        else:
            output = b_norm

        # print network info:
        stride_str = str([stride, stride])
        filter_size = str(weight.get_shape().as_list())
        output_size = str(output.get_shape().as_list())
        param_num = '{0:<10.6f}M'.format((in_depth * kernel_size[0] * kernel_size[1] * depth + depth)/(10**6))
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_size, filter_size, stride_str, padding, output_size, param_num
            )
        print(md_str)
        # print('{0:<15s}| {1:<25s}| {2:<25s}| {3:<25s}| {4:<15s}'.format(
        #     name, input_size, filter_size, output_size, param_num))
        return output


@add_arg_scope
def deconv2d(name, x, depth, kernel_size, stride=1, padding='SAME', is_training=True, weight_init='he_normal', b=0.01, activation=None, norm=True):
    input_size = str(x.get_shape().as_list())
    with tf.variable_scope(name):
        in_depth = x.get_shape().as_list()[-1]
        if weight_init == 'he_normal':
            w_init = tf.initializers.he_normal()
        elif weight_init == 'glorot_uniform':
            w_init = tf.initializers.glorot_uniform()
        elif weight_init == 'glorot_normal':
            w_init = tf.initializers.glorot_normal()
        else:
            w_init = tf.initializers.glorot_normal()
        weight = tf.get_variable(name='weight', shape=[kernel_size[0], kernel_size[1], in_depth, depth], dtype=tf.float32,
                                 initializer=w_init)
        bias = tf.constant_initializer(b)
        conv = tf.keras.layers.Conv2DTranspose(depth, kernel_size, strides=stride, padding=padding,
                                               kernel_initializer=w_init, bias_initializer=bias)(x)
        if norm is True:
            b_norm = tf.layers.batch_normalization(
                conv,
                axis=-1,
                training=is_training,
                trainable=True,
                momentum=0.99,
                renorm_momentum=0.9
            )
        else:
            b_norm = conv

        if activation == 'relu':
            output = tf.nn.relu(b_norm)
        elif activation == 'leaky_relu':
            output = tf.nn.leaky_relu(b_norm)
        else:
            output = b_norm

        # print network info:
        stride_str = str([stride, stride])
        filter_size = str(weight.get_shape().as_list())
        output_size = str(output.get_shape().as_list())
        param_num = '{0:<10.6f}M'.format((in_depth * kernel_size[0] * kernel_size[1] * depth + depth)/(10**6))
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_size, filter_size, stride_str, padding, output_size, param_num
            )
        print(md_str)
        return output


@add_arg_scope
def maxpool2d(name, x, kernel_size, stride=1, padding='SAME'):
    with tf.variable_scope(name):
        output = tf.nn.max_pool(x,
                                ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, stride, stride, 1],
                                padding=padding
                                )
        # print info
        input_info = str(x.get_shape().as_list())
        pool_size = str(kernel_size)
        stride_str = str([stride, stride])
        out_info = str(output.get_shape().as_list())
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_info, pool_size, stride_str, padding, out_info, ""
            )
        print(md_str)
        return output

@add_arg_scope
def avgpool2d(name, x, kernel_size, stride=1, padding='SAME'):
    with tf.variable_scope(name):
        output = tf.nn.avg_pool(x,
                                ksize=[1, kernel_size[0], kernel_size[1], 1],
                                strides=[1, stride, stride, 1],
                                padding=padding
                                )
        # print info
        input_info = str(x.get_shape().as_list())
        pool_size = str(kernel_size)
        stride_str = str([stride, stride])
        out_info = str(output.get_shape().as_list())
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_info, pool_size, stride_str, padding, out_info, ""
            )
        print(md_str)
    return output

@add_arg_scope
def upsampling2d(name, x, kernel_size, method='nearest'):
    output = tf.keras.layers.UpSampling2D(kernel_size, interpolation=method, name=name)(x)
    # print info
    input_info = str(x.get_shape().as_list())
    pool_size = str(kernel_size)
    out_info = str(output.get_shape().as_list())
    md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
        name, input_info, pool_size, "", "", out_info, ""
    )
    print(md_str)

    return output

def flatten(name, x):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()[1:]
        xflat = tf.reshape(x, [-1, np.product(x_shape)])

        # print info
        input_info = str(x.get_shape().as_list())
        out_info = str(xflat.get_shape().as_list())
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_info, "", "", "", out_info, ""
            )
        print(md_str)
        return xflat

@add_arg_scope
def dense(name, x, units, is_training=True, weight_init='he_normal', activation=None, norm=True):
    in_size = x.get_shape().as_list()[-1]
    input_info = str(x.get_shape().as_list())
    b_value = 0.01
    if weight_init == 'he_normal':
        w_init = tf.initializers.he_normal()
    elif weight_init == 'glorot_uniform':
        w_init = tf.initializers.glorot_uniform()
    elif weight_init == 'glorot_normal':
        w_init = tf.initializers.glorot_normal()
    else:
        w_init = tf.initializers.glorot_normal()

    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', shape=[in_size, units], dtype=tf.float32,
                                 initializer=w_init)
        bias = tf.Variable(np.ones(shape=units)*b_value, dtype=tf.float32, name='bias')
        output = tf.nn.xw_plus_b(x, weight, bias)
        if norm is True:
            b_norm = tf.layers.batch_normalization(
                output,
                axis=-1,
                training=is_training,
                trainable=True,
                momentum=0.99,
                renorm_momentum=0.9
            )
        else:
            b_norm = output

        # print info:
        weight_info = str(weight.get_shape().as_list())
        out_info = str(output.get_shape().as_list())
        param_info = '{0:<10.6f}M'.format((in_size*units + units)/(10**6))
        md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
                name, input_info, weight_info, "", "", out_info, param_info
            )
        print(md_str)
        if activation == 'relu':
            return tf.nn.relu(b_norm)
        elif activation == 'softmax':
            return tf.nn.softmax(b_norm)
        elif activation == 'leaky_relu':
            return tf.nn.leaky_relu(b_norm)
        else:
            return b_norm

@add_arg_scope
def dropout(name, funct_output, prob=0.5, is_training=False):
    with tf.variable_scope(name):
        if is_training is True:
            return tf.nn.dropout(funct_output, rate=prob)
        else:
            return funct_output


def reshape(name, x, shape):
    with tf.variable_scope("Reshape"):
        output = tf.reshape(x, shape=shape, name=name)
    # print info
    input_info = str(x.get_shape().as_list())
    out_info = str(output.get_shape().as_list())
    md_str = '{0:20s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s} | {6:20s}'.format(
        name, input_info, "", "", "", out_info, ""
    )
    print(md_str)
    return output


def each_loss(end_points, y_target, is_training, aux=False):
    if aux is False:
        each_ce = -tf.reduce_sum(
            y_target * tf.log(tf.clip_by_value(end_points['Prediction'], 1e-10, 1.0)),
            reduction_indices=[1],
            name='each_loss_predict'
        )
    else:
        if is_training is True:
            each_prediction = -tf.reduce_sum(
                y_target * tf.log(tf.clip_by_value(end_points['Prediction'], 1e-10, 1.0)),
                reduction_indices=[1],
                name='each_loss_predict'
            )
            each_aux_prediction = -tf.reduce_sum(
                y_target * tf.log(tf.clip_by_value(end_points['AuxPrediction'], 1e-10, 1.0)),
                reduction_indices=[1],
                name='each_loss_auxpredict'
            )
            each_ce = each_prediction + 0.3*each_aux_prediction
        else:
            each_ce = -tf.reduce_sum(
                y_target * tf.log(tf.clip_by_value(end_points['Prediction'], 1e-10, 1.0)),
                reduction_indices=[1],
                name='each_loss_predict'
            )
    return each_ce


def avg_loss(each_ce):
    avg_ce = tf.reduce_mean(each_ce, name='avg_loss')
    return avg_ce


def accuracy(prediction, y_target):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_target, 1))
    accu = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    return accu


def model_summary(keras_model, param_dict=None, valid_acc_dict=None, print_out=True, save_dir=None):
    str_list = []
    s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
        'Name', 'Input_shape', 'Kernel size', 'Strides', 'Padding', 'Output_shape'
    )
    str_list.append(s)
    str_list.append("-"*121)
    for l in keras_model.layers:
        if l.name.split('_')[0] == 'conv2d':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'conv2d' and l.name.split('_')[1] == 'transpose':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.kernel._keras_shape), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)

        elif l.name.split('_')[0] == 'batch':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", ""
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'activation':
            str0 = l.output.name.split("/")
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                str0[0]+"/"+str0[-1], "", "", "", "", "")
            str_list.append(s)
        elif l.name.split('_')[0] == 'leaky':
            s = '{0:25s}'.format(
                l.name)
            str_list.append(s)
        elif (l.name.split('_')[0] == 'max') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'avg') and (l.name.split('_')[1] == 'pooling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.pool_size), str(l.strides), l.padding, str(l.output_shape)
            )
            str_list.append(s)
        elif (l.name.split('_')[0] == 'up') and (l.name.split('_')[1] == 'sampling2d'):
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", str(l.size), "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'flatten' or l.name.split('_')[0] == 'reshape':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), "", "", "", str(l.output_shape))
            str_list.append(s)
        elif l.name.split('_')[0] == 'dense':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(
                l.name, str(l.input_shape), str(l.units), "", "", str(l.output_shape)
            )
            str_list.append(s)
        elif l.name.split('_')[0] == 'dropout':
            s = '{0:25s} | {1:25s} | {2:20s} | {3:8s} | {4:8s} | {5:20s}'.format(l.name, str(l.rate), "", "", "", "")
            str_list.append(s)
    str_list.append("-"*121)

    if param_dict:
        for key in sorted(param_dict.keys()):
            s = key + ": " + str(param_dict[key])
            str_list.append(s)
        str_list.append("-" * 121)

    if valid_acc_dict:
        str_list.append(str(len(valid_acc_dict)) + "-FOLD VALIDATION ACCURACY")
        acc = []
        for key in sorted(valid_acc_dict.keys()):
            acc.append(valid_acc_dict[key])
            s = key + ": " + "{0:7.4f}%".format(100*valid_acc_dict[key])
            str_list.append(s)
        mean_acc = np.mean(np.asarray(acc))
        str_list.append("AVG: {0:7.4f}%".format(100*mean_acc))

    if print_out:
        for s in str_list:
            print(s)
    if save_dir:
        with open(save_dir, 'w') as f:
            for s in str_list:
                f.write(s + "\n")

    return str_list


def N_Fold_Validate(n_splits, num_data):
    splits = n_splits + 1
    id_max = num_data - 1
    seq = np.linspace(0, id_max, splits, endpoint=True, dtype=int)
    nf_list = []
    for i in range(n_splits):
        start = seq[i]
        end = seq[i+1]
        nf_list.append(np.arange(start, end).tolist())

    nf_list_f = []
    for id, nf in enumerate(nf_list):
        temp = nf_list.copy()
        temp.pop(id)
        train = []
        for t in temp:
            train = train + t
        nf_list_f.append([train, nf])

    return nf_list_f


