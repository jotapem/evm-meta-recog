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

class vggDataset():#torch.utils.data.Dataset):
    def __init__(self, data_path, subdir, verbose=False):
        self.verbose = verbose
        
        self.path_root = data_path
        self.path_data = os.path.join(self.path_root, subdir)
        self.path_descriptors = os.path.join('.', 'vgg_descriptors', subdir)
        

        self.cpp_recognizer = face_recognition.FaceRecognition()
        self.cpp_recognizer.ConfigCNN('face_core/data/face_recognition_cnn_model.dat')
        self.cpp_recognizer.LoadLandmarkModel('face_core/data/landmark_full.dat')
    
        self.cpp_detector = face_detection.FaceDetection()
        self.cpp_detector.SetLandmarkExtractor("face_core/data/landmark_full.dat")


        self.data = None

        # gets label->pathes mapping
        print("Building anno mapping")
        self._set_anno()
        print("Mapping done")



    def _img_proc(self, image_path):
        loaded_image = cv2.imread(image_path)

        detections, landmarks, poses = self.cpp_detector.DetectFacesAndLandmarks(loaded_image, 1.0, False, True, 0.0)

        if len(detections)==1: # only when the labeling is clear
            descriptor = np.squeeze(self.cpp_recognizer.getSingleDescriptorCNN(loaded_image, detections[0])[0].reshape(-1, 128))
            
            return {'image': loaded_image, 'descriptor': descriptor}
        else:
            return None
        
    def _store_descriptor(self, label, image_id) -> bool : 
        """
        Given a labeled identified image, attempts to extract a single face detection and its descriptors
        Might skip this process (case the descriptor seem to be saved already or a single face descriptor cannot be extracted)
        Returns whether the (label,image_id) tuple yields a descriptor
        """

        
        image_fullpath = os.path.join(self.path_data, label, image_id)
        pickle_outpath = os.path.join(self.path_descriptors, label, image_id)

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

    def _set_anno(self):
        """
        """

        self.anno = {}
        
        for r,d,f in os.walk(self.path_data):
            if (r == self.path_data):
                continue

            label = r.split('/')[-1]
            #label = int(r.split('/')[-1][1:])
            images = f

            #if len(images) >= self.person_sample_min: # unnecessary cut here
            images.sort()
            self.anno[label] = images                

            #print(label, len(images))

            
    def get_descriptors(self, persons_limit=None, person_sample_min=0, person_sample_max=None):
        """
        Process raw_images in self.path_data generating descriptors, storing them on self.path_descriptors
        """

        labels = sorted(self.anno.keys()) # could be shuffle or smte
        print("%d labels" % (len(labels)))
        

        # compute descriptors and store them
        descriptors = {}
        people_counter = 0
        
        for label in labels:
            person_data = []
            
            # creates label dir 
            my_mkdir(os.path.join(self.path_descriptors, label))
            
            i_ids = self.anno[label]

            #np_label = np.array([label]).astype('int')
            sample_counter = 0
            for i_id in i_ids:
                if self._store_descriptor(label, i_id):
                    person_data += [os.path.join(self.path_descriptors, label, i_id)]
                    sample_counter += 1
                    if person_sample_max is not None and sample_counter >= person_sample_max:
                        break

            if len(person_data) >= person_sample_min:
                descriptors.update({label: person_data})
                people_counter += 1
                if persons_limit is not None and people_counter >= persons_limit:
                    break
                
        print("Processing/storage done")
        return descriptors

    def get_training_data(self, samples_train:int, samples_test:int=0, persons_limit:int=None) -> list:
        """
        Input interface is how many persons (possibly how many there is), how many samples in training (gallery control) and how many samples for test (possibly none)
        Output interface is a list of (X,Y) numpy stuff
        """
        n_samples = (samples_train + samples_test)
        
        descriptors = self.get_descriptors(persons_limit=persons_limit, person_sample_min=n_samples, person_sample_max=n_samples)

        X_train, Y_train, X_test, Y_test = [], [], [], []
        for label, pathes in descriptors.items():
            pathes_train, pathes_test = pathes[:samples_train], pathes[-samples_test:]

            X_train += list(map(lambda x: load_pickle(x), pathes_train))
            X_test += list(map(lambda x: load_pickle(x), pathes_test))
            Y_train += list(map(lambda x: np.array(label), pathes_train))
            Y_test += list(map(lambda x: np.array(label), pathes_test))
    
        return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))

