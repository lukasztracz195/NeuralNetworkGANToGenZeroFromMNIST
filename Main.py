from keras.models import Sequential
from sympy.integrals.rubi.utility_function import Flatten
from tensorflow.python.keras.layers import Dense, BatchNormalization, Reshape, Dropout, UpSampling2D, Conv2DTranspose, \
    Activation, Conv2D, LeakyReLU
from tensorflow.python.keras.optimizers import RMSprop

from gan.DCGAN import DCGAN
from timing.ElapsedTimer import ElapsedTimer

dcgan = DCGAN()

### UTWORZENIE WARSTWY DLA GENERATORA ####
generator = Sequential()
dropout = 0.4
depth = 64 + 64 + 64 + 64
dim = 7
# In: 100
# Out: dim x dim x depth
generator.add(Dense(units=dim * dim * depth, input_dim=100))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Reshape((dim, dim, depth)))
generator.add(Dropout(dropout))

# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
generator.add(UpSampling2D())  # NIE ROZUMIEM
generator.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))

generator.add(UpSampling2D())  # NIE ROZUMIEM
generator.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))

generator.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))

# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
generator.add(Conv2DTranspose(1, 5, padding='same'))
generator.add(Activation('sigmoid'))
generator.summary()

dcgan.generator = generator

### UTWORZENIE SIECI DLA DYSKRYMINATORA ###

discriminator = Sequential()
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
discriminator.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(filters=256, kernel_size=5, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Conv2D(filters=2048, kernel_size=5, strides=1, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(dropout))

# Out: 1-dim probability
discriminator.add(Flatten())
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))
discriminator.summary()
dcgan.discriminator = discriminator
#################################################
dcgan.optimizer_for_discriminator_model = RMSprop(lr=0.0002, decay=6e-8)
dcgan.optimizer_for_adversarial_model = RMSprop(lr=0.0001, decay=3e-8)
dcgan.loss_for_discriminator_model = 'binary_crossentropy'
dcgan.loss_for_adversarial_model = 'binary_crossentropy'
dcgan.metrics_for_discriminator_model = ['accuracy']
dcgan.metrics_for_adversarial_model = ['accuracy']

# trenowanie
timer = ElapsedTimer()
dcgan.train(train_X=None, train_Y=None, batch_size=256, save_interval=10)
timer.elapsed_time()

dcgan.plot_images(fake=True)
dcgan.plot_images(fake=False, save2file=True)
