import os

import cv2
import numpy as np
import torch
import torchvision
import pickle

from face_core.wrappers.python.lib import face_recognition
from face_core.wrappers.python.lib import face_detection

def my_mkdir(p):
    if not os.path.isdir(p):
        os.mkdir(p)

def load_pickle(p):
    ret = None
    with open(p, 'rb') as pickle_in:
        ret = pickle.load(pickle_in)
    return ret

class vggDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, subdir, verbose=False, persons_limit=None):
        self.verbose = verbose
        self.persons_limit = persons_limit
        
        self.path_root = data_path
        self.path_data = os.path.join(self.path_root, subdir)
        self.path_descriptors = os.path.join('.', 'vgg_descriptors', subdir)
        

        self.cpp_recognizer = face_recognition.FaceRecognition()
        self.cpp_recognizer.ConfigCNN('face_core/data/face_recognition_cnn_model.dat')
        self.cpp_recognizer.LoadLandmarkModel('face_core/data/landmark_full.dat')
    
        self.cpp_detector = face_detection.FaceDetection()
        self.cpp_detector.SetLandmarkExtractor("face_core/data/landmark_full.dat")

        
        self._generate_descriptors()



    def _img_proc(self, image_path):
        loaded_image = cv2.imread(image_path)

        detections, landmarks, poses = self.cpp_detector.DetectFacesAndLandmarks(loaded_image, 1.0, False, True, 0.0)

        if len(detections)==1: # only when the labeling is clear
            descriptor = np.squeeze(self.cpp_recognizer.getSingleDescriptorCNN(loaded_image, detections[0])[0].reshape(-1, 128))
            
            return {'image': loaded_image, 'descriptor': descriptor}
        else:
            return None
        
    def _store_descriptor(self, label, image_id, train=True) -> bool : 
        """
        Given a labeled identified image, attempts to extract a single face detection and its descriptors
        Might skip this process (case the descriptor seem to be saved already or a single face descriptor cannot be extracted)
        Returns whether the (label,image_id) tuple yields a descriptor
        """

        
        image_fullpath = os.path.join(self.path_data, label, image_id)
        pickle_outpath = os.path.join(self.path_descriptors, ('train' if train else 'test'), label, image_id)

        if os.path.isfile(pickle_outpath):
            if self.verbose: print("Already processed descriptor %s" % (os.path.join(label, image_id)))
            return True
        else:
            proc = self._img_proc(image_fullpath)

            if proc is not None:
                descriptor = proc['descriptor']

                with open(pickle_outpath, 'wb') as pickle_out:
                    pickle.dump(descriptor, pickle_out)
                    print("Saved descriptor at %s" % (pickle_outpath))
                return True
            else:
                return False


    
    def _generate_descriptors(self):
        """
        Process raw_images in self.path_data generating descriptors, storing them on self.path_descriptors
        """
                
        # gets label->pathes mapping

        print("Building anno mapping")
        l_train, l_test = 10, 5
        self.anno = {}
        
        i=0
        for r,d,f in os.walk(self.path_data):
            if (r == self.path_data):
                continue

            label = r.split('/')[-1]
            #label = int(r.split('/')[-1][1:])
            images = f

            if len(images) >= l_train+l_test+3:
            #if len(images) >= 10 and len(images) <= 500:
                images.sort()
                
                #print(label, len(images))

                self.anno[label] = images[:l_train+l_test+3]
                ''' # using raw image ids and labels
                self.anno[label] = []

                for img_path in images:
                    img_fullpath = os.path.join(r, img_path)
                    self.anno[label].append(img_fullpath)
                '''

            i += 1
            if (self.persons_limit and i>=self.persons_limit): break

        print("Mapping done")
        #print(self.anno)
        print(sorted(self.anno.keys()))



        # compute descriptors and store them
        self.train_data, self.test_data = [], []
        
        for label in self.anno:
            # creates label dir for test and train
            my_mkdir(os.path.join(self.path_descriptors, 'train', label))
            my_mkdir(os.path.join(self.path_descriptors, 'test', label))
            exit
            
            i_ids = self.anno[label]

            #np_label = np.array([label]).astype('int')
            for i_id in i_ids[:l_train]:
                if self._store_descriptor(label, i_id):
                    self.train_data += [{'label': label, 'descriptor_path': os.path.join(self.path_descriptors, 'train', label, i_id)}]
                
            for i_id in i_ids[l_test:]:
                if self._store_descriptor(label, i_id, train=False):
                    self.test_data += [{'label': label, 'descriptor_path': os.path.join(self.path_descriptors, 'test', label, i_id)}]
                
        print("Processing/storage done")


    
    def _set_descriptors_paths(self):
        """
        Initializes self.train_images and self.test_images, as a list of pickle paths
        """
        #self.train_images = []
        for r,d,f in os.walk(os.path.join(self.path_descriptors, 'train')):
            for i in f:
                fullpath = os.path.join(r, i)
                #self.train_images += [fullpath]

        # that f dangerous copy n paste (test / train)
        #self.test_images = []
        for r,d,f in os.walk(os.path.join(self.path_descriptors, 'test')):
            for i in f:
                fullpath = os.path.join(r, i)
                #self.test_images += [fullpath]

    
    def _get_classifier_data(self, fullpath, train=False):
        """
        Loads X, y given data in self.(train/test)_data
        """
        mode_path = 'train' if train else 'test'
        
        label = fullpath.split('/')[-2]
        descriptor = load_pickle(fullpath)

        return descriptor, label
        
                
    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        #X, y = self.train_images[index]['descriptor'], self.train_images[index]['label']
        #X, y = torch.DoubleTensor(X), int(y)#torch.LongTensor(y)
        X, y = self._get_classifier_data(self.train_images[index])

        #print(X.shape, y.shape)
        
        return X, y
        
