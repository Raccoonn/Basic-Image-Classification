import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    - Add more layers to handle Reshaping input images

    - use tensorflow_addons to add normalization layers

"""





class classification_CNN:
    def __init__(self, input_shape, activation='relu', optimizer='adam'):
        """
        Create and compile model
            - Still need to add customization for parameters in each layer
            - Currently all sizes and neuron numbers are hard coded
        """
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(10))

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()




    def save_model_weights(self, filename):
        """
        Save weights with given filename
        """
        self.model.save_weights(filename, overwrite=True, save_format=None, options=None)



    def load_model_weights(self, filename):
        """
        Load weights from the given filename
        """
        self.model.load_weights(filename, by_name=False, skip_mismatch=False, options=None)





    def train_model(self, train_images, train_labels, epochs, validation_data):
        """
        Train and evaluate the model
        """
        history = self.model.fit(train_images, train_labels, epochs=epochs, validation_data=validation_data)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_images, test_labels = validation_data
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)




    def make_predictions(self, test_images):
        """
        Feedforward on the preloaded test data and return predictions
            - The model can take in an entire np.array and predict on all data
        """
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(test_images)

        return predictions




    def show_results(self, predictions, test_images, test_labels, class_names,
                     num_rows=5, num_cols=3, randomize=True):
        """
        Shows results for the given predictions.  Generates a plot that visualizes
        the results for a random sample from the calculated predictions

            - Define num_rows, num_columns to format the plot
            - Note extra indexing for labels, remember labels are list of list 
              containing an int
        """
        def plot_image(i, j, predictions, true_label, img):
            true_label, img = true_label[j], img[j]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.imshow(img, cmap=plt.cm.binary)

            predicted_label = np.argmax(predictions)
            if predicted_label == true_label:
                color = 'blue'
            else:
                color = 'red'

            plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                                100*np.max(predictions),
                                                class_names[true_label[0]]),
                                                color=color)


        def plot_value_array(i, j, predictions, true_label):
            true_label = true_label[j][0]
            plt.grid(False)
            plt.xticks(range(10))
            plt.yticks([])
            thisplot = plt.bar(range(10), predictions, color="#777777")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions)

            thisplot[predicted_label].set_color('red')
            thisplot[true_label].set_color('blue')


        # Plot a random selection from the predictions
        # Using the test_data as initially defined, pre-flattening
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):

            # Whether or not to select random elements from training set
            if randomize == True:
                j = np.random.randint(0, len(predictions))
            else:
                j = i

            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(i, j, predictions[j], test_labels, test_images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(i, j, predictions[j], test_labels)

        plt.tight_layout()
        plt.show()

