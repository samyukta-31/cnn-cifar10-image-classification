from Preprocessing.format import Processing
from keras.losses import categorical_crossentropy
from CNN_build.cnn import CNN
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns

class Testandval(Processing, CNN):
    def __init__(self):
        super().__init__()
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.x_train, self.x_test, self.train_labels, self.test_labels = self.train_test_build()

    def testing(self):
        model_3 = self.layering()
        epochs = 20
        opt = 'adam'

        # Optimizing model to reduce loss and fitting it with the training data
        model_3.compile(loss = categorical_crossentropy, optimizer = opt, metrics = ['accuracy'])
        run_3 = model_3.fit(self.x_train, self.train_labels, epochs=epochs, validation_data=(self.x_test, self.test_labels))

        # Printing loss and accuracy scores of final model
        score = model_3.evaluate(self.x_test, self.test_labels)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Predicting labels for the given classes based on the above model and generating confusion matrix
        pred_labels_3 = model_3.predict(self.x_test)
        return pred_labels_3, run_3
        
    
    def evaluate(self):

        pred_labels_3, run_3 = self.testing()

        matrix_3 = metrics.confusion_matrix(self.test_labels.argmax(axis=1), pred_labels_3.argmax(axis=1))

        # Tranforming confusion matrix into dataframe and assigning labels as columns 
        cf_df_3 = pd.DataFrame(matrix_3, index = [i for i in self.class_names],
                        columns = [i for i in self.class_names])

        # Plotting; Rows show the actual class of a repetition and columns show the classifier's prediction
        plt.figure(figsize = (10,10))
        ax = sns.heatmap(cf_df_3, annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        print(metrics.classification_report(self.test_labels.argmax(axis=1),
                                            pred_labels_3.argmax(axis=1),target_names=self.class_names))

        # Visualize accuracy history
        plt.plot(run_3.history['accuracy'])
        plt.plot(run_3.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
        plt.xlim([0, 20])
        plt.show()

        # Visualize loss history
        plt.plot(run_3.history['loss'])
        plt.plot(run_3.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'], loc='upper left')
        plt.xlim([0, 20])
        plt.show()