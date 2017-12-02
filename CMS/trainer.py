import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from CMS.dataset.face_detection_dataset import FaceDetectionDataset
from CMS.temp_cms import CMSRCNN
from CMS.transforms.mean_subtract_transform import MeanSubtract
from CMS.transforms.scale_transform import Scale


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


faces_image_path = "./data/WIDER_train/images/"
json_path = "./data/wider_face_split/wider_face_train_bbx_gt.json"

transformer = Compose([MeanSubtract(), Scale()])
dataset = FaceDetectionDataset(json_file=json_path, root_dir=faces_image_path, transform=transformer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
net = CMSRCNN()
net = nn.DataParallel(net).cuda()
optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001)
epochs = 100
print("Begin Training")
batch_loss, total_loss = 0, 0
batches = 0

for epoch in range(epochs):
    for batch, sample in enumerate(dataloader, 0):

        inputs = sample["image"]
        bboxes = sample["bboxes"].numpy()
        info = sample["scale"]
        bboxes_count = sample["bboxes_count"].numpy()

        temp = []
        # np.empty((len(bboxes), bboxes_count, 5))

        for i in range(len(bboxes)):
            temp.append(bboxes[:, :bboxes_count[i]])

        # bboxes = bboxes.reshape((g, -1, 5))

        # temp = np.empty((0, 0))
        #
        # bboxes = bboxes[:, bboxes != -1].reshape((len(bboxes), -1, 5))
        #
        # for i in range(len(bboxes)):
        #     bboxes[i,:,:] = bboxes[i, :bboxes_count[i], :]
        # for i in range(len(bboxes)):
        #     temp = np.append(temp, [[bboxes[i, :bboxes_count[i], :]]])
        # bboxes = np.vsplit(bboxes, bboxes_count)

        bboxes = temp
        # bboxes = np.array(temp, dtype=object)
        inputs = Variable(inputs.cuda())
        # bboxes = Variable(bboxes.cuda())

        optimizer.zero_grad()
        # def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        outputs = net(inputs, info.numpy(), bboxes)
        loss = net.module.loss + net.module.rpn.loss
        # loss = criterion(outputs, bboxes)
        loss.backward()
        optimizer.step()
        batch_loss += loss.data[0]
        batches += 1
        if batch != 0 and batch % 30 == 0:
            print("Epoch = {}, Batch = {}, Error = {}".format(epoch, batch, batch_loss))
            total_loss += batch_loss
            batch_loss = 0
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    })
    total_loss = total_loss / batches
    print("epoch finished with loss = {}".format(total_loss))
    total_loss, batches = 0, 0