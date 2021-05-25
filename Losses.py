from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
import tensorflow as tf


class KLLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="KLLoss"):
        super().__init__(reduction=reduction, name=name)

    def call(self, z_mean, z_logvar):
        z_mean = ops.convert_to_tensor_v2(z_mean)
        z_logvar = math_ops.cast(z_logvar, z_mean.dtype)
        kl_loss = 1 + z_logvar - math_ops.square(z_mean) - math_ops.exp(z_logvar)
        kl_loss = -0.5 * math_ops.reduce_sum(kl_loss, axis=-1)

        return kl_loss
