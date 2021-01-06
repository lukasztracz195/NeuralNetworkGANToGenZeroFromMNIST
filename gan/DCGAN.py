import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop

import timing


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel_color=1):

        self.__img_rows = img_rows
        self.__img_cols = img_cols
        self.__channel = channel_color
        self.__discriminator = None  # discriminator
        self.__generator = None  # generator
        self.__adversarial_model = None  # adversarial model // model przeciwności
        self.__discriminator_model = None  # discriminator model // model dyskrymiancji
        self.__optimizer_for_discriminator_model = None
        self.__loss_for_discriminator_model = None
        self.__metrics_for_discriminator_model = list()
        self.__optimizer_for_adversarial_model = None
        self.__loss_for_adversarial_model = None
        self.__metrics_for_adversarial_model = list()
        self.__path_to_save_images = None
        self.__train_X = None
        self.__train_Y = None

    @property
    def img_rows(self):
        return

    @img_rows.setter
    def img_rows(self, value):
        self.__img_rows = value

    @property
    def __img_cols(self):
        return self.__img_cols

    @__img_cols.setter
    def __img_cols(self, value):
        self.__img_cols = value

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, value):
        self.__channel = value

    @property
    def discriminator(self):
        return self.__discriminator

    @discriminator.setter
    def discriminator(self, value):
        self.__discriminator = value

    @property
    def generator(self):
        return self.__generator

    @generator.setter
    def generator(self, value):
        self.__generator = value

    @property
    def loss_for_discriminator_model(self):
        return self.__loss_for_discriminator_model

    @loss_for_discriminator_model.setter
    def loss_for_discriminator_model(self, value):
        self.__loss_for_discriminator_model = value

    @property
    def optimizer_for_discriminator_model(self):
        return self.__optimizer_for_discriminator_model

    @optimizer_for_discriminator_model.setter
    def optimizer_for_discriminator_model(self, value):
        self.__optimizer_for_discriminator_model = value

    @property
    def metrics_for_discriminator_model(self):
        return self.__metrics_for_discriminator_model

    @metrics_for_discriminator_model.setter
    def metrics_for_discriminator_model(self, value):
        self.__metrics_for_discriminator_model = value

    @property
    def loss_for_adversarial_model(self):
        return self.__loss_for_adversarial_model

    @loss_for_adversarial_model.setter
    def loss_for_adversarial_model(self, value):
        self.__loss_for_adversarial_model = value

    @property
    def optimizer_for_adversarial_model(self):
        return self.__optimizer_for_adversarial_model

    @optimizer_for_adversarial_model.setter
    def optimizer_for_adversarial_model(self, value):
        self.__optimizer_for_adversarial_model = value

    @property
    def metrics_for_adversarial_model(self):
        return self.__metrics_for_adversarial_model

    @metrics_for_adversarial_model.setter
    def metrics_for_adversarial_model(self, value):
        self.__metrics_for_adversarial_model = value

    @property
    def path_to_save_images(self):
        return self.__path_to_save_images

    @path_to_save_images.setter
    def path_to_save_images(self, value):
        self.__path_to_save_images = value

    @property
    def discriminator_model(self):
        if self.__discriminator_model:
            return self.__discriminator_model
        self.__discriminator_model = Sequential()
        self.__discriminator_model.add(self.discriminator(), trainable=True)
        self.__discriminator_model.compile(loss=self.__loss_for_discriminator_model,
                                           optimizer=self.__optimizer_for_discriminator_model,
                                           metrics=self.__metrics_for_discriminator_model)
        return self.__discriminator_model

    @property
    def adversarial_model(self):
        if self.__adversarial_model:
            return self.__adversarial_model
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.__adversarial_model = Sequential()
        self.__adversarial_model.add(self.generator())
        self.__adversarial_model.add(self.discriminator(), trainable=False)
        self.__adversarial_model.compile(loss=self.__loss_for_adversarial_model,
                                         optimizer=self.__optimizer_for_adversarial_model,
                                         metrics=self.__metrics_for_adversarial_model)
        return self.__adversarial_model

    def train(self, train_X, train_Y, train_steps=10, batch_size=256, save_interval=0):
        self.__train_X = train_X
        self.__train_Y = train_Y
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

        for i in range(train_steps):
            start_d = timing.time()
            images_train = self.__train_X[np.random.randint(0, self.__train_X.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            end_d = timing.time()
            duration_d = end_d - start_d
            start_g = timing.time()
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.__adversarial_model.train_on_batch(noise, y)
            end_g = timing.time()
            duration_g = end_g - start_g

            log_message = "%d: [Discriminator loss: %f, acc: %f | duration: %f s]" % (
                i, d_loss[0], d_loss[1], duration_d)
            log_message = "%s  [Generator loss: %f, acc: %f | duration: %f s]" % (
                log_message, a_loss[0], a_loss[1], duration_g)
            print(log_message)
            if save_interval > 0 and noise_input:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, fake=True, samples=noise_input.shape[0], noise=noise_input,
                                     step=(i + 1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = "mnist_%d.png" % step
        images = self.generator.predict(noise)
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        if not fake:
            i = np.random.randint(0, self.__train_X.shape[0], samples)
            images = self.__train_X[i, :, :, :]
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.__img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(self.__path_to_save_images + filename)
            plt.close('all')
        else:
            plt.show()
