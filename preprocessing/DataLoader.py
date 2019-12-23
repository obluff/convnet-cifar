from functools import reduce
import numpy as np
import pickle
import torch

class DataLoader:
    """Class that can be run to load the training data for Cifar-10 and runs preprocessing + data augmentation


        argument
            input_dir - directory of your inputs
            data_augmentation_functions - a list of functions that can be applied to an individual image
                                          for augmentation. if left blank, images won't be augmented
            peprocessing_functions - a list of functions that can be applied to an entire
                                     np array of images that can be used for dataprep and will
                                     be run on both the TRAIN and TEST SET, if left blank
                                     images won't be augmented
        returns
            torch datasets for:
                training data
                test data
                label encodings

        usage:
            data_loader = DataLoader(input_dir, data_augmentation_functions, preprocessing_functions)
            train_data, test_data, label_encodings = data_loader.execute()
    """
    def __init__(self, input_dir, data_augmentation_functions=[], data_prep_functions=[]):
        self.input_dir = input_dir
        self.data_augmentation_functions = data_augmentation_functions
        self.data_prep_functions = data_prep_functions


    def execute(self):
        """
            argument
                self - input dir, data augmentation functions, data prep functions
            returns
                torch datasets for:
                    training data
                    test data
                    label encodings
       """
        train_data, train_labels, test_data, test_labels = self.load_data(self.input_dir)

        train_data = self.append_augmentations(train_data, self.data_augmentation_functions)
        train_labels = np.tile(train_labels, len(self.data_augmentation_functions) + 1)
        train_data, test_data = self.preprocess_images(train_data, self.data_prep_functions), \
                                self.preprocess_images(test_data, self.data_prep_functions)


        train_data, test_data = self.make_dataset(train_data, train_labels), \
                                self.make_dataset(test_data, test_labels)

        label_encodings = self.unpickle_and_encode(''.join([self.input_dir, '/batches.meta']))['label_names']

        return train_data, test_data, label_encodings

    def make_dataset(self, images, labels):

        """
            argument
                images - np array of images
                labels - np array of labels
            returns
                torch data set of the images + labels
                tranposes the images away fom PIL format into one ready for training
        """
        return torch.utils.data.TensorDataset(torch.Tensor(images.transpose(0, 3, 1,2)),
                                              torch.LongTensor(labels))


    def preprocess_images(self, images, functions):
        """
            argument
                images - np array of images
                functions - functions to perform AFTER data augmentation, on
                            both the train and the test set, for example
                            normalization
            returns
                images with preprocessing functions performed to them
        """
        if not functions:
            return images

        for func in functions:
            images = func(images)
        return images

    def append_augmentations(self, images, functions):

        """
            argument
                images - np array of images
                functions - list of functions to augment the data with
            returns
                original image data appended with augmentation

            can be improvd by allowing one to pass in a parameter
            to only augment a random sample of a portion of the array
            for each augmentation
            """
        if not functions:
            return images

        applied_images = map(lambda f: np.array([f(x) for x in images]), functions)
        return np.append(images, reduce(lambda x, y: np.append(x, y, axis=0), applied_images), axis=0)


    def load_data(self, input_dir):
        """
            argument
                - input_dir: path to input directory
            returns
                - np arrays of train data, test data and there respective labels
        """
        training_files = map(lambda x: self.import_image(''.join([input_dir, '/', 'data_batch_', str(x)])), \
                                                                        range(2, 6))

        train_data, train_labels = reduce(lambda x, y: (np.append(x[0], y[0], axis=0),\
                                                        np.append(x[1], y[1], axis=0)), training_files)

        test_data, test_labels = self.import_image('/'.join([input_dir, 'test_batch']))

        train_data, test_data

        return train_data, train_labels, \
               test_data, test_labels

    def import_image(self, path):
        """
            argument
                - path: path to serialized image data
            returns
                - images and labels

            transposes loaded array to PIL format so we can do preprocessing
        """
        raw_image_data = self.unpickle_and_encode(path)

        images, labels = raw_image_data['data'], raw_image_data['labels']
        images = images.reshape((images.shape[0], 3, 32, 32)).transpose(0,2,3,1)

        return images, labels

    def unpickle_and_encode(self, file):
        """
            argument
                - file: path to serialized file
            returns
                - dictionary containing the data with keys
                  encoded as utf-8
        """
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
        return {str(k, 'utf-7'): v for k, v in dic.items()}
