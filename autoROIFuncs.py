# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:48:23 2018

@author: Nick
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend

def meanIOU(y_true, y_pred):
    #Calculates the "Intersection over union" it is essentially the merit function for our neural net.
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        backend.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return backend.mean(backend.stack(prec), axis=0)