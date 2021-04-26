from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F

from torchreid.utils import open_all_layers, open_specified_layers
from ..engine import Engine
from torchreid.losses import TripletLoss, CrossEntropyLoss, SupConLoss
from torchvision import transforms
from tools import compute_mean_std


class ImageSupConEngine(Engine):

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=10,
            weight_x=1,
            weight_r=0,
            weight_arc=0,
            weight_center=0,
            weight_scl=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageSupConEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        # self.weight_t = weight_t
        # self.weight_x = weight_x
        self.weight_scl = weight_scl

        # self.criterion_t = TripletLoss(margin=margin)
        # self.criterion_x = CrossEntropyLoss(
        #     num_classes=self.datamanager.num_train_pids,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth
        # )

        self.criterion_scl = SupConLoss()

    def transform(self, x):
        mean = [0.485, 0.456, 0.406]  # imagenet mean
        std = [0.229, 0.224, 0.225]  # imagenet std
        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        #print(train_transform(x).shape)
        return [train_transform(x), train_transform(x)]

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda(non_blocking=True)
            pids = pids.cuda(non_blocking=True)


        features = self.extract_features(imgs, pids)
        # loss1_x = self.compute_loss(self.criterion_x, outputs1, pids)
        # loss1_t = self.compute_loss(self.criterion_t, features1, pids)
        loss_scl = self.compute_loss(self.criterion_scl, features, pids)

        # outputs2, features2 = self.model2(imgs)
        # loss2_x = self.compute_loss(self.criterion_x, outputs2, pids)
        # loss2_t = self.compute_loss(self.criterion_t, features2, pids)
        #
        # loss1_ml = self.compute_kl_div(
        #     outputs2.detach(), outputs1, is_logit=True
        # )
        # loss2_ml = self.compute_kl_div(
        #     outputs1.detach(), outputs2, is_logit=True
        # )
        #loss = 0
        loss = loss_scl
        # loss1 += loss1_t * self.weight_t
        # loss1 += loss1_ml * self.weight_ml

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'loss1_scl': loss_scl.item(),

        }

        return loss_dict

    # def extract_features(self, input, labels):
    #     trans_imgs = self.transform(input)
    #     #print(trans_imgs)
    #     #x1 = torch.from_numpy(x1).long()
    #     #x2 = torch.from_numpy(x2).long()
    #     #print(trans_imgs[0].shape)
    #
    #     images = torch.cat([trans_imgs[0], trans_imgs[1]], dim=0)
    #     print(images.shape)
    #     bsz = labels.shape[0]
    #     features = self.model(images)
    #     #print(features)
    #     f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    #     #print(f1)
    #     #print(f2)
    #     features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    #
    #     return features
