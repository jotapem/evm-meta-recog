from torch.utils.data.dataset import Dataset

from recognition.image_feature_extractor import ImageFeatureExtractor


class FaceRecognitionDataset(Dataset):
    def __init__(self, faces):
        self.faces = faces

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        return self.faces[index]
