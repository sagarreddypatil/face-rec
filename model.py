from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

class Model:
    def __init__(self, num_classes = 3, input_shape=(128, 128, 3)):
        self.model = Sequential()
        self.model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.summary()

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        print("Model Initilazed")
    def train(self, X, y, X_valid, y_valid, batch_size = 32, epochs = 25):
        X  = X / 255
        X_valid = X_valid / 255

        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))
    def predict(self, X):
        X = X / 255
        return self.model.predict(X)
    def save(self, path_to_folder = "/"):
        self.model.save(path_to_folder + 'model.h5')
        print("Model Saved")
    def load(self, path_to_model = "/model.h5"):
        self.model = load_model(path_to_model)
        print("Model loaded from disk")