from keras.datasets import cifar10
from keras.utils import to_categorical


class Processing:
    def train_test_build(self):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        num_classes = 10

        # Summarizing loaded dataset
        print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
        print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))

        # Reshaping the train and test input data
        train_images = train_images.reshape(50000, 3072)
        test_images = test_images.reshape(10000, 3072)
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # Normalize pixel values to be between 0 and 1
        train_images /= 255
        test_images /= 255

        # Printing the final hare of train and test samples
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')

        # Converting class vectors to binary class matrix
        train_labels = to_categorical(train_labels, num_classes)
        test_labels = to_categorical(test_labels, num_classes)


        # Reshaping the train and test image matrices to the correct shape for the CNN model
        img_rows, img_cols = 32, 32
        x_train = train_images.reshape(train_images.shape[0], img_rows, img_cols, 3)
        x_test = test_images.reshape(test_images.shape[0], img_rows, img_cols, 3)

        return x_train, x_test, train_labels, test_labels


