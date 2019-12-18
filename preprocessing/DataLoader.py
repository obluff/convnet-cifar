from functools import reduce
import numpy as np
import pickle
import torch

class DataLoader:
    def __init__(self, input_dir, data_augmentation_functions=[], data_prep_functions=[]):
        self.input_dir = input_dir
        self.data_augmentation_functions = data_augmentation_functions
        self.data_prep_functions = data_prep_functions
        
    
    def execute(self):
        """Returns
            
           train_data, train_labels, test_data, test_labels, 
       
       """
        train_data, train_labels, test_data, test_labels = self.load_data(self.input_dir)
        
        train_data = self.append_augmentations(train_data, self.data_augmentation_functions)
        train_labels = np.tile(train_labels, len(self.data_augmentation_functions) + 1)
        train_data, test_data = self.preprocess_images(train_data, self.data_prep_functions), \
                                self.preprocess_images(test_data, self.data_prep_functions)
        print(train_data.shape)
        print(test_data.shape)
        print(train_labels.shape)
        train_data, test_data = self.make_dataset(train_data, train_labels), \
                                self.make_dataset(test_data, test_labels)
        
        label_encodings = self.unpickle_and_encode(''.join([self.input_dir, '/batches.meta']))['label_names']
        
        return train_data, test_data, label_encodings
    
    def make_dataset(self, data, labels):
        """Takes in Numpy array and returns a Tensor Dataset"""
        return torch.utils.data.TensorDataset(torch.Tensor(data.transpose(0, 3, 1,2)), 
                                              torch.LongTensor(labels))
    
   
    def preprocess_images(self, images, functions):
        """Simple forloop for function composition for preprocessing"""
        for func in functions:
            images = func(images)
        return images
    
    def append_augmentations(self, images, functions):
        """appends image_transformations to original array"""
        applied_images = map(lambda f: np.array([f(x) for x in images]), functions)
        return np.append(images, reduce(lambda x, y: np.append(x, y, axis=0), applied_images), axis=0)
                      
       
    def load_data(self, input_dir):
        """Takes in an input training directory and returns 

            train_data, train_labels
            test_data, test_labels"""
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
