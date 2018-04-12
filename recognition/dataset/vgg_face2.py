import os

import cv2
import numpy as np
import torch
import pickle

from face_core.wrappers.python.lib import face_recognition
from face_core.wrappers.python.lib import face_detection

class vggDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.path_root = data_path
        self.path_train = os.path.join(data_path, 'train')

        self.cpp_recognizer = face_recognition.FaceRecognition()
        self.cpp_recognizer.ConfigCNN('face_core/data/face_recognition_cnn_model.dat')
        self.cpp_recognizer.LoadLandmarkModel('face_core/data/landmark_full.dat')
    
        self.cpp_detector = face_detection.FaceDetection()
        self.cpp_detector.SetLandmarkExtractor("face_core/data/landmark_full.dat")


        # gets label->pathes mapping
        self.anno = {}
        #max_people=3
        i=0
        for r,d,f in os.walk(self.path_train):
            if (r == self.path_train):
                continue

            label = int(r.split('/')[-1][1:])
            images = f

            #if len(images) >= 15:
            if len(images) >= 10 and len(images) <= 500:
                images.sort()
                
                #print(label, len(images))

                self.anno[label] = []
                for img_path in images:
                    img_fullpath = os.path.join(r, img_path)
                    self.anno[label].append(img_fullpath)

            i += 1
            #if (i>=max_people): break

        print("Mapping done")
        #print(self.anno)
        print(sorted(self.anno.keys()))

        # load processed images in array
        self.train_images = []
        self.test_images = []
        for label in self.anno:
            i_paths = self.anno[label]

            np_label = np.array([label]).astype('int')
            for i_path in i_paths[:-5]:
                proc = self._img_proc(i_path)
                if proc is not None:
                    proc.update({'label': label})
                    self.train_images.append(proc)
                
            for i_path in i_paths[-5:]:
                proc = self._img_proc(i_path)
                if proc is not None:
                    proc.update({'label': label})
                    self.test_images.append(proc)
        print("Processing done")

        with open('vgg_train.pkl', 'wb') as pickle_out:
            pickle.dump(self.train_images, pickle_out)
        with open('vgg_test.pkl', 'wb') as pickle_out:
            pickle.dump(self.test_images, pickle_out)
        print("Pickling done")


    def _img_proc(self, image_path):
        loaded_image = cv2.imread(image_path)

        detections, landmarks, poses = self.cpp_detector.DetectFacesAndLandmarks(loaded_image, 1.0, False, True, 0.0)

        if len(detections)==1: # only when the labeling is clear
            descriptor = np.squeeze(self.cpp_recognizer.getSingleDescriptorCNN(loaded_image, detections[0])[0].reshape(-1, 128))
            
            return {'image': loaded_image, 'descriptor': descriptor}
        else:
            return None
        

        
        
    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        X, y = self.train_images[index]['descriptor'], self.train_images[index]['label']
        X, y = torch.DoubleTensor(X), int(y)#torch.LongTensor(y)

        #print(X.shape, y.shape)
        
        return X, y
        
