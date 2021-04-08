import torch
import torchreid
from contrastive_learner import ContrastiveLearner
#from torchvision import models




def Contrastive(num_classes, loss='triplet', pretrained=True, **kwargs):

    ConEncoder = torchreid.models.build_model(
        name='vit_timm_diet',
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained

    )

    #torchreid.utils.load_pretrained_weights(ConEncoder, '/home/danish/deep-person-reid/scripts/log/model/model.pth.tar-112')

    learner = ContrastiveLearner(
        ConEncoder,
        image_size=224,
        hidden_layer='head',
        # layer name where output is hidden dimension. this can also be an integer specifying the index of the child
        project_hidden=True,  # use projection head
        project_dim=128,  # projection head dimensions, 128 from paper
        use_nt_xent_loss=False,  # the above mentioned loss, abbreviated
        temperature=0.1,  # temperature
        augment_both=True  # augment both query and key
    )
    return learner

# opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
#
# def sample_batch_images():
#     return torch.randn(20, 3, 256, 256)
#
# for _ in range(100):
#     images = sample_batch_images()
#     loss = learner(images)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
