from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization


class CNN:
    def layering(self):
        # Generating layers of the CNN
        img_rows, img_cols = 32, 32
        model_3 = Sequential()
        model_3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',
                        input_shape=(img_rows, img_cols, 3),strides=1,padding='same'))
        model_3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_3.add(BatchNormalization())
        model_3.add(MaxPooling2D(pool_size=(2, 2)))
        model_3.add(Dropout(0.2))

        model_3.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_3.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_3.add(BatchNormalization())
        model_3.add(MaxPooling2D(pool_size=(2, 2)))
        model_3.add(Dropout(0.2))

        model_3.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_3.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model_3.add(BatchNormalization())
        model_3.add(MaxPooling2D(pool_size=(2, 2)))
        model_3.add(Dropout(0.2))

        model_3.add(Flatten())
        model_3.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model_3.add(Dropout(0.2))
        model_3.add(Dense(10, activation='softmax'))
        model_3.summary()

        return model_3