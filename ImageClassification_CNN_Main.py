
from CNN_importFunctions import *
from ImageClassification_CNN import classification_CNN



"""
Main driver function for tf CNN.

    - Currently capable of loading and training on CIFAR, MNIST Digits, and MNIST Fashion

"""


if __name__ == '__main__':

    set_choice = input('\nCIFAR, Digits, or Fashion?      ')
    load_data_from_file = input('\nLoad data from file?       ')


    if set_choice == 'CIFAR':
        print('\n\nLoading Training and Testing data...\n\n')

        if load_data_from_file in ('Y', 'y'):
            train_filename, test_filename = 'CIFAR_train_data.pickle', 'CIFAR_test_data.pickle'
            (train_images, train_labels), (test_images, test_labels), class_names = import_saved_data(train_filename, test_filename, 'CIFAR')
        else:
            (train_images, train_labels), (test_images, test_labels), class_names = load_CIFAR()
            

    elif set_choice == 'Digits':
        print('\n\nLoading Training and Testing data...\n\n')

        if load_data_from_file in ('Y', 'y'):
            train_filename, test_filename = 'MNIST_digits_train_data.pickle', 'MNIST_digits_test_data.pickle'
            (train_images, train_labels), (test_images, test_labels), class_names = import_saved_data(train_filename, test_filename, 'Digits')
        else:
            (train_images, train_labels), (test_images, test_labels), class_names = load_MNIST_digits()


    elif set_choice == 'Fashion':
        print('\n\nLoading Training and Testing data...\n\n')

        if load_data_from_file in ('Y', 'y'):
            train_filename, test_filename = 'MNIST_fashion_train_data.pickle', 'MNIST_fashion_test_data.pickle'
            (train_images, train_labels), (test_images, test_labels), class_names = import_saved_data(train_filename, test_filename, 'Fashion')
        else:
            (train_images, train_labels), (test_images, test_labels), class_names = load_MNIST_fashion()


    show_sample_data(train_images, train_labels, class_names)


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
        Model.train_model(train_images, train_labels, 40, (test_images, test_labels))
        weightSave = input('\n\nSave weights to file?       ')
        if weightSave in ('Y', 'y'):
            save_filename = input('\n\nInput filename to save weights:      ')
            Model.save_model_weights(save_filename)
        


    input('\n\nPress Enter to check predictions...\n\n')

    if set_choice == 'Digits':
        if input('\nLoad user handwriting images?       ') in ('Y', 'y'):
            filepath = input('\nFilepath for user digits 1-9:     ')
            test_images, test_labels = import_images(filepath, [str(n) for n in range(10)], '.png')

    # show_sample_data(test_images, test_labels, class_names)

    predictions = Model.make_predictions(test_images)

    Model.show_results(predictions, test_images, test_labels, class_names,
                       num_rows=5, num_cols=3, randomize=True)
