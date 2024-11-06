import torchvision
import torch

from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from config import NUM_CLASSES as nc

def create_model(num_classes=91, min_size=640, max_size=640):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights='DEFAULT'
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    model.transform.min_size = [min_size] if isinstance(min_size, int) else min_size #min_size
    model.transform.max_size = max_size
    return model

if __name__ == '__main__':
    model = create_model(nc)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
