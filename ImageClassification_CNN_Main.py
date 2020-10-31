
from CNN_importFunctions import *
from ImageClassification_CNN import classification_CNN



"""
Completed CNN.  Includes driver function with
command line interface.  Includes save/load 
functions. Supports other datasets.


Need to add:
    - More support for input images, use keras.layers.Resize
"""




if __name__ == '__main__':

    set_choice = input('\nCIFAR or MNIST?      ')
    load_data_from_file = input('\nLoad data from file?       ')

    
    if set_choice == 'CIFAR':
        print('\n\nLoading Training and Testing data...\n\n')

        if load_data_from_file in ('Y', 'y'):
            train_filename, test_filename = 'CIFAR_train_data.pickle', 'CIFAR_test_data.pickle'
            (train_images, train_labels), (test_images, test_labels) = import_CIFAR(train_filename, test_filename)
        else:
            (train_images, train_labels), (test_images, test_labels) = load_CIFAR()
            
        class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])



    elif set_choice == 'MNIST':
        print('\n\nLoading Training and Testing data...\n\n')

        if load_data_from_file in ('Y', 'y'):
            train_filename, test_filename = 'MNIST_digits_train_data.pickle', 'MNIST_digits_test_data.pickle'
            (train_images, train_labels), (test_images, test_labels) = import_MNIST_digits(train_filename, test_filename)
        else:
            (train_images, train_labels), (test_images, test_labels) = load_MNIST_digits()

        class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']



    # Initialize and compile model
    input_shape = test_images[0].shape
    Model = classification_CNN(input_shape, activation='relu', optimizer='SGD')

    load_model_weights = input('\nLoad weights from file?       ')
    if load_model_weights in ('Y', 'y'):
        Model.load_model_weights(input('\nFilename to load weights:     '))
        print('\n\nModel successfully loaded...\n\n')

    else:
        input('\n\nPress Enter to begin training...\n\n')
        # Train model and save weights
        Model.train_model(train_images, train_labels, 10, (test_images, test_labels))
        Model.save_model_weights(input('\nFilename to save weights:        '))
        


    input('\n\nPress Enter to check predictions...\n\n')

    if set_choice == 'MNIST':
        if input('\nLoad user handwriting images?       ') in ('Y', 'y'):
            filepath = input('\nFilepath for user digits 1-9:     ')
            test_images, test_labels = import_images(filepath, [str(n) for n in range(10)], '.png')

    show_sample_data(test_images, test_labels, class_names)

    predictions = Model.make_predictions(test_images)

    Model.show_results(predictions, test_images, test_labels, class_names,
                       num_rows=5, num_cols=3, randomize=True)
