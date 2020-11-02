from PIL import Image
import PIL.ImageOps
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




def make_RGB(images):
    """ Converts a grayscale image set into a RGB tensor """
    img_x, img_y = images[0].shape
    images = tf.constant(images.reshape(len(images), img_x, img_y, 1))
    images = tf.image.grayscale_to_rgb(images)

    return np.array(images, dtype=np.float)





def load_MNIST_digits():
    """
    Load in MNIST digits set and convert to RGB tensors
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']


    # Change labels to follow CIFAR format --> list of lists containing single int each
    train_labels = np.array([[l] for l in train_labels])
    test_labels = np.array([[l] for l in test_labels])

    train_images = make_RGB(train_images)
    test_images = make_RGB(test_images)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Pickle list of lists containg each image and label pair
    train_data = [(image, label) for image, label in zip(train_images, train_labels)]
    test_data = [(image, label) for image, label in zip(test_images, test_labels)]

    with open('MNIST_digits_train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('MNIST_digits_test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (train_images, train_labels), (test_images, test_labels), class_names





def load_MNIST_fashion():
    """
    Load in MNIST fashion set and convert to RGB tensors
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Change labels to follow CIFAR format --> list of lists containing single int each
    train_labels = np.array([[l] for l in train_labels])
    test_labels = np.array([[l] for l in test_labels])

    train_images = make_RGB(train_images)
    test_images = make_RGB(test_images)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Pickle list of lists containg each image and label pair
    train_data = [(image, label) for image, label in zip(train_images, train_labels)]
    test_data = [(image, label) for image, label in zip(test_images, test_labels)]

    with open('MNIST_fashion_train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('MNIST_fashion_test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (train_images, train_labels), (test_images, test_labels), class_names







def load_CIFAR():
    """
    Download and save the CIFAR data set to file
        - Using a pickled dictionary for storage
        - labels are keys to corresponding image array
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    # Scale pixel data
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Pickle list of lists containng each image and label pair
    train_data = [(image, label) for image, label in zip(train_images, train_labels)]
    test_data = [(image, label) for image, label in zip(test_images, test_labels)]

    with open('CIFAR_train_data.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('CIFAR_test_data.pickle', 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return (train_images, train_labels), (test_images, test_labels), class_names





def import_images(parent_folder, filenames, file_extension, H=28, W=28):
    """
    Import images from the given folder in the  list of filenames.
    
    Returns test_images in 2d np.array after converting [R,G,B] value to int
    """
    # Concatenate strings for full file paths
    filenames = [parent_folder + name + file_extension for name in filenames]

    raw_images = []
    for f in filenames:
        image = Image.open(f).resize((H,W))
        image = PIL.ImageOps.invert(image)
        raw_images.append(np.asarray(image))

    # Format --> list of lists containing an integer each
    test_images = np.array(raw_images)
    test_labels = np.array([[N] for N in range(len(raw_images))])       # images are loaded sequentially
        
    return test_images, test_labels




def import_data(filename):
    """
    Load in new training data from pickle file
        - Format is list of lists, containing each image/label pair
        - Label format is list of lists containg 1 int each
    """
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    images, labels = [], []
    for line in data:
        images.append(line[0])
        labels.append(line[1])

    return np.array(images), np.array(labels)




def import_saved_data(train_filename, test_filename, data_type):
    train_images, train_labels = import_data(train_filename)
    test_images, test_labels = import_data(test_filename)

    if data_type == 'CIFAR':
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    elif data_type == 'Digits':
        class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

    elif data_type == 'Fashion':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    return (train_images, train_labels), (test_images, test_labels), class_names





def show_sample_data(images, labels, class_names):
    """
    Show a plot of sample data taken from the set
    """
    plt.figure(figsize=(10,10))
    plt.title('Selection of Sample Data')
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[labels[i][0]])
    plt.show()






