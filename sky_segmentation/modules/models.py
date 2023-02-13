import torch
import torchvision
import torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule


class PretrainedSegmentationModule(torch.nn.Module):
    def __init__(self, encoder_weights_path, decoder_weights_path):
        super().__init__()
        net_encoder = ModelBuilder.build_encoder(
            arch="resnet50dilated",
            fc_dim=2048,
            weights=encoder_weights_path,
        )
        net_decoder = ModelBuilder.build_decoder(
            arch="ppm_deepsup",
            fc_dim=2048,
            num_class=150,
            weights=decoder_weights_path,
            use_softmax=True,
        )
        crit = torch.nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).eval()
        self.segmentation_module = segmentation_module

    def forward(self, x, output_size):
        scores = self.segmentation_module(x, segSize=output_size)
        _, pred = torch.max(scores, dim=1)
        return pred


# a wrapper for the segmentation module to output the sky class only
class SkyPretrainedSegmentationModule(torch.nn.Module):
    def __init__(self, encoder_weights_path, decoder_weights_path, sky_index=2):
        super().__init__()
        self.sky_index = sky_index
        net_encoder = ModelBuilder.build_encoder(
            arch="resnet50dilated",
            fc_dim=2048,
            weights=encoder_weights_path,
        )
        net_decoder = ModelBuilder.build_decoder(
            arch="ppm_deepsup",
            fc_dim=2048,
            num_class=150,
            weights=decoder_weights_path,
            use_softmax=True,
        )
        crit = torch.nn.NLLLoss(ignore_index=-1)
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).eval()
        self.segmentation_module = segmentation_module

    def forward(self, x, image_size):
        scores = self.segmentation_module(x, segSize=image_size)
        _, pred = torch.max(scores, dim=1)
        return torch.where(pred == self.sky_index, 1, 0)


def get_inference_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
