
from PIL import Image
import PIL.ImageOps
import tensorflow as tf
from ImageClassification_basicNN import classificationNN
import numpy as np




def download_fashion():
    """
    Load in mnist Fashion dataset and define class names
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_data, train_labels, test_data, test_labels, class_names




def download_digits():
    """
    Load in mnist handwritten digits and define class names
    """
    digits_mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = digits_mnist.load_data()

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

    return train_data, train_labels, test_data, test_labels, class_names






def import_images(parent_folder, filenames, file_extension, H=28, W=28):
    """
    Import images from the given folder in the  list of filenames.
    
    Returns test_images in 2d np.array after converting [R,G,B] value to int
    """
    def getIfromRGB(rgb):
        R, G, B = rgb
        RGBint = ( (R<<16) + (G<<8) + B)
        return RGBint

    # Concatenate strings for full file paths
    filenames = [parent_folder + name + file_extension for name in filenames]

    raw_images = []
    for f in filenames:
        image = Image.open(f).resize((H,W))
        image = PIL.ImageOps.invert(image)
        raw_images.append(image)

    test_labels = [N for N in range(len(raw_images))]       # images are loaded sequentially

    test_images = []

    for image in raw_images:
        img = np.asarray(image)
        img_mat = np.zeros((H,W))
        for i in range(H):
            for j in range(W):
                rgb_val = getIfromRGB(img[i,j][:3])
                img_mat[i,j] = rgb_val
        
        test_images.append(img_mat)
        
    return test_images, test_labels






if __name__ == '__main__':

    # train_images, train_labels, test_images, test_labels, class_names = download_fashion()
    # train_images, train_labels, test_images, test_labels, class_names = download_digits()


    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']



    # Setup model
    # Model = classificationNN(train_images, train_labels, test_images, test_labels, class_names)

    # Normalize data
    # Model.normalize_data(train_filename='Fashion_train', test_filename='Fashion_test')
    # Model.normalize_data(train_filename='Digits_train', test_filename='Digits_test')


    # Empty model if loading data
    Model = classificationNN(None, None, None, None, class_names)

    # Load data if pre-normalized
    # Model.load_normalized_data('Fashion_train', 'Fashion_test')
    Model.load_normalized_data('Digits_train', 'Digits_test')




    # Compile the model with specified N for hidden neurons, and N for output neurons
    Model.compile_model(128, 10, activation='tanh')

    # Train model for given N epochs
    Model.train_model(10)

    # Load in different digit dataset to test on
    # new_test_images, new_test_labels = import_images('Michaela Digits/', [str(n) for n in range(10)], '.png')
    # Model.load_new_test_data(new_test_images, new_test_labels, test_filename='Kyle_test')


    # Make predictions on test data
    predictions = Model.make_predictions()




    # Show a sample of the results from the test data
    Model.show_results(Model, predictions, num_rows=5, num_cols=2, randomize=False)


    input('\n\nPress <Enter> to quit.')



