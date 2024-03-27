from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
import keras
import numpy as np
import cv2

class Model:
    def __init__(self, input_shape):
        self.model = Sequential()
        
        self.model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu',input_shape=input_shape))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(256, activation = "relu")) #Fully connected layer
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(60, activation = "relu")) #Fully connected layer
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(12, activation = "softmax")) #Classification layer or output layer


        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def fit(self, training_batch_generator, testing_batch_generator, epochs):
        self.model.fit_generator(
            generator=training_batch_generator,
            steps_per_epoch = len(training_batch_generator),
            epochs=epochs,
            verbose=1,
            validation_data=testing_batch_generator,
            validation_steps=len(testing_batch_generator)
        )

    def save(self, filename):
        self.model.save(filename)

class CustomGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        return np.array([
                cv2.imread('../data/processed/' + str(file_name))
                for file_name in batch_x]) /255, np.array(batch_y)
