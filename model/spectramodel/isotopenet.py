import numpy as np
import tensorflow as tf
from .spectra_normalizer import SpectraNormalizer
from .csv_loader import csv_loader as loader
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models, optimizers, losses, metrics, regularizers, backend as K

class IsotopeNet():

    def __init__(self, lr=5e-4, model_type='base', n_classes=3):

        assert model_type in ['base']
        assert n_classes >= 2 and n_classes <= 4

        self.spectra_size = 8000
        self.model_type = model_type
        self.model = None

        self.n_classes = n_classes
        self.output_neurons = 1 if self.n_classes == 2 else self.n_classes
        self.activation = 'sigmoid' if self.n_classes == 2 else 'softmax'
        self.loss = losses.BinaryCrossentropy() if self.n_classes == 2 else losses.CategoricalCrossentropy()
        
        self.lr = lr
        self.optimizer = optimizers.Adam(learning_rate=self.lr, amsgrad=True)
        self.kernel_regularizer = regularizers.l2(l=0.05)
        self.metrics = [tf.keras.metrics.BinaryAccuracy()] if self.n_classes == 2 else [tf.keras.metrics.CategoricalAccuracy()]

        if self.n_classes == 2:
            self.metrics.append(tf.keras.metrics.Precision())
            self.metrics.append(tf.keras.metrics.Recall())
        else:
            for i in range(self.n_classes):
                self.metrics.append(tf.keras.metrics.Precision(class_id=i))
                self.metrics.append(tf.keras.metrics.Recall(class_id=i))

        self.__build__()

    def compile(self):

        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

    def train(self, generator, config):
        
        training_results = self.model.fit(generator, **config)
        self.metrics_names = self.model.metrics_names
        return training_results

    def predict(self, generator, config):

        return self.model.predict(generator, **config)

    def raw_predict(self, x, config=None):

        if config is not None:
            return self.model.predict(x, **config)
        else:
            return self.model.predict(x)

    def evaluate(self, generator, config):

        return self.model.evaluate(generator, **config)

    def save(self, save_path):

        self.model.save_weights(str(save_path))

    def load(self, save_path):

        self.model.load_weights(str(save_path))

    def summary(self):

        self.model.summary()

    def csv_loader(self, root):

        return loader(root)

    def data_loader(self, files):

        spectra_loader = lambda s: np.loadtxt(files.in_path / '{}.txt'.format(s)).reshape((-1, 1, 1))

        return spectra_loader

    def preprocessing(self, **kwargs):

        def __preprocess_spectra__(x):

            normalizer = SpectraNormalizer(x)

            if 'baseline_median' in kwargs and kwargs['baseline_median'] is not None:
                normalizer.baseline_median(kwargs['baseline_median'])

            if 'smoothing_moving_average' in kwargs and kwargs['smoothing_moving_average'] is not None:
                normalizer.smoothing_moving_average(kwargs['smoothing_moving_average'])

            if 'normalize_tic' in kwargs and kwargs['normalize_tic'] is not None and kwargs['normalize_tic'] == True:
                normalizer.normalize_tic()

            x = normalizer.get()

            return x

        return __preprocess_spectra__

    def raw_model(self):

        return self.model

    def __build__(self):

        if self.model_type == 'simple_cnn':
            self.__simple_cnn_model__()

        if self.model_type == 'simple_rescnn':
            self.__simple_rescnn_model__()

        if self.model_type == 'base':
            self.__base_model__()

    def __simple_cnn_model__(self):
        
        input_spectra = layers.Input(shape=(self.spectra_size, 1, 1))

        kernel = (5, 1)
        stride_1 = (1, 1)
        stride_3 = (3, 1)
        padding = 'same'

        x = layers.Conv2D(8, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(input_spectra)
        x = layers.Conv2D(8, kernel, strides=stride_3, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(16, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(16, kernel, strides=stride_3, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(32, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(32, kernel, strides=stride_3, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(64, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(64, kernel, strides=stride_3, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(128, kernel, strides=stride_3, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(128, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Conv2D(256, kernel, strides=stride_1, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(self.output_neurons, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Activation(self.activation)(x)

        self.model = models.Model(input_spectra, x)


    def __simple_rescnn_model__(self):

        input_spectra = layers.Input(shape=(self.spectra_size, 1, 1))

        x = self.__residual_layer__(input_spectra, 16, 5, 1)
        x = self.__residual_layer__(x, 32, 5, 3)
        x = self.__residual_layer__(x, 32, 5, 1)
        x = self.__residual_layer__(x, 64, 5, 3)
        x = self.__residual_layer__(x, 64, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 128, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 128, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 128, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 128, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 128, 5, 1)
        x = self.__residual_layer__(x, 128, 5, 3)
        x = self.__residual_layer__(x, 256, 5, 3)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(self.output_neurons, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Activation(self.activation)(x)

        self.model = models.Model(input_spectra, x)

    def __base_model__(self):

        input_spectra = layers.Input(shape=(self.spectra_size, 1, 1))

        x = self.__residual_layer__(input_spectra, 8, 3, 1)
        x = self.__residual_layer__(x, 8, 3, 5)
        x = self.__residual_layer__(x, 8, 3, 1)
        x = self.__residual_layer__(x, 1, 3, 3)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LocallyConnected2D(1, (5, 1), kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.output_neurons, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.Activation(self.activation)(x)

        self.model = models.Model(input_spectra, x)

    def __residual_layer__(self, x, filters, kernel, stride):

        kernel = (kernel, 1)
        stride = (stride, 1)
        padding = 'same'

        identity = x

        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel, strides=(1, 1), padding=padding, kernel_regularizer=self.kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel, strides=stride, padding=padding, kernel_regularizer=self.kernel_regularizer)(x)

        if stride != (1, 1):
            identity = layers.Conv2D(filters, (1, 1), strides=stride, padding=padding, kernel_regularizer=self.kernel_regularizer)(identity)

        x = layers.Add()([identity, x])

        return x
