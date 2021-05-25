import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from VAEModel import VAEModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Create 4 virtual GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

EPOCHS = 50
BATCH_SIZE = 256
ALL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_dim = x_train.shape[1] ** 2
x_train = np.reshape(x_train, [-1, input_dim])
x_test = np.reshape(x_test, [-1, input_dim])
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(input_dim).batch(ALL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(ALL_BATCH_SIZE)

latent_dim = 32

with strategy.scope():
    vae_model = VAEModel(input_dim, latent_dim)
    vae_model.model_summary(input_shape=input_dim)
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_train = tf.keras.metrics.Mean()
    loss_test = tf.keras.metrics.Mean()

with strategy.scope():
    def train_step(inputs, global_batch_size):
        x = inputs
        # print(x)
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z, x_re = vae_model(x, training=True)
            # VAE loss
            vae_loss = vae_model.vae_loss(x, x_re, z_mean, z_logvar, global_batch_size=global_batch_size, reduction=tf.keras.losses.Reduction.NONE)

        # update gradient
        grads = tape.gradient(vae_loss, vae_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae_model.trainable_variables))

        loss_train(vae_loss)
        return vae_loss

    def test_step(inputs, global_batch_size):
        x = inputs
        z_mean, z_logvar, z, x_re = vae_model(x, training=True)
        # VAE loss
        vae_loss = vae_model.vae_loss(x, x_re, z_mean, z_logvar, global_batch_size=global_batch_size, reduction=tf.keras.losses.Reduction.NONE)
        loss_test(vae_loss)
        return vae_loss

with strategy.scope():
    @tf.function
    def distributed_train_step(dataset_inputs, global_batch_size):
        per_replica_losses = strategy.run(train_step,
                                          args=(dataset_inputs, global_batch_size))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs, global_batch_size):
        per_replica_losses = strategy.run(test_step,
                                          args=(dataset_inputs, global_batch_size))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    for epoch in range(EPOCHS):

        for train in train_dataset:
            distributed_train_step(train, ALL_BATCH_SIZE)

        for test in test_dataset:
            distributed_test_step(test, ALL_BATCH_SIZE)

        print("Epoch {}/{} | Train loss: {} | Validation loss: {}".format(epoch+1, EPOCHS, loss_train.result().numpy(), loss_test.result().numpy()))
        loss_train.reset_states()
        loss_test.reset_states()

decoded_imgs = vae_model.predict(x_test)[3]

n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
