import numpy as np
import tensorflow as tf
from math import *

def build_model():
    inputs = tf.keras.layers.Input((224,224,3),name = 'input')

    model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet", input_tensor = inputs)
    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name = 'pooling')(x)
    outputs = tf.keras.layers.Dense(1,activation = 'sigmoid',name = 'outputs')(x)

    model = tf.keras.Model(inputs=inputs , outputs =outputs)

    return model