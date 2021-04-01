import torch
from byol_pytorch import BYOL
# from torchvision import models
import torchvision
from torch import nn
import kornia

from torchreid.models import vittimm


def byol_vit(num_classes, loss='softmax', pretrained=True, **kwargs):
    vit = vittimm(2048, loss=loss, pretrained=pretrained, **kwargs)

    augment_fn = nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip()
    )

    augment_fn2 = nn.Sequential(
        kornia.augmentation.RandomHorizontalFlip(),
        kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
    )

    learner = BYOL(
        vit,
        image_size=224,
        hidden_layer='pre_logits',
        projection_size=num_classes,  # the projection size
        projection_hidden_size=4096,  # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay=0.99,
        # the moving average decay factor for the target encoder, already set at what paper recommends
        augment_fn=augment_fn,
        augment_fn2=augment_fn2,
    )

    print(learner)
    return learner

    # opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

#
# def sample_unlabelled_images():
#     return torch.randn(20, 3, 256, 256)
#
#
#
#
# for _ in range(100):
#     images = sample_unlabelled_images()
#     loss = learner(images)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     learner.update_moving_average()  # update moving average of target encoder
#
# # save your improved network
# torch.save(resnet.state_dict(), './improved-net.pt')
