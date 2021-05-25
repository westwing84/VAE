import tensorflow as tf
from tensorflow.keras import backend as K
from Losses import KLLoss


class VAEModel(tf.keras.Model):

    def __init__(self, input_dim, latent_dim):
        super(VAEModel, self).__init__(self)

        self.input_dim = input_dim
        # Encoder
        self.encoder1 = tf.keras.layers.Dense(256, activation="relu")
        self.encoder2 = tf.keras.layers.Dense(64, activation="relu")

        # latent representation
        self.latent_mean = tf.keras.layers.Dense(latent_dim)
        self.latent_logvar = tf.keras.layers.Dense(latent_dim)
        self.latent = tf.keras.layers.Lambda(self.sampling, output_shape=(latent_dim,))

        # Decoder
        self.decoder1 = tf.keras.layers.Dense(64, activation="relu")
        self.decoder2 = tf.keras.layers.Dense(256, activation="relu")
        self.reconstructed_output = tf.keras.layers.Dense(input_dim, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        # Encoder
        x = self.encoder2(self.encoder1(inputs))
        z_mean = self.latent_mean(x)
        z_logvar = self.latent_logvar(x)
        z = self.latent([z_mean, z_logvar])

        # Decoder
        x = self.decoder2(self.decoder1(z))
        x_re = self.reconstructed_output(x)

        return z_mean, z_logvar, z, x_re

    def vae_loss(self, x, x_re, z_mean, z_logvar, global_batch_size, reduction=tf.keras.losses.Reduction.NONE):
        # Kullback-Leibler loss
        kl_loss_metrics = KLLoss(reduction=reduction)
        kl_loss = tf.nn.compute_average_loss(kl_loss_metrics(z_mean, z_logvar), global_batch_size=global_batch_size) / global_batch_size

        # Reconstruction loss
        # mse = tf.keras.losses.MeanSquaredError(reduction=reduction)
        mse = tf.keras.losses.BinaryCrossentropy(reduction=reduction)
        reconstruction_loss = tf.nn.compute_average_loss(mse(x, x_re), global_batch_size=global_batch_size)

        final_loss = reconstruction_loss + kl_loss

        return final_loss

    def sampling(self, inputs):
        # Reparametrization Trick
        z_mean, z_logvar = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), seed=5)
        return z_mean + K.exp(0.5 * z_logvar) * epsilon

    def model_summary(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        model.summary()
        # del x, model
