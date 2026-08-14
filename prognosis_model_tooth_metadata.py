import sys

import torch
import torch.nn as nn

sys.path.append("../Segmentation")

from UNet import UNet3D


class PrognosisModel(nn.Module):
    """
    3D CBCT prognosis classifier using the encoder of UNet3D.

    `channels` must match the segmentation checkpoint architecture
    when pretrained weights are used.

    Tooth metadata:
        [mandibular, anterior, premolar, molar]
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        dropout=0.3,
        metadata_dim=4,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2, 2),
        prelu=False,
    ):
        super().__init__()

        self.channels = tuple(
            int(x)
            for x in channels
        )

        self.metadata_dim = int(
            metadata_dim
        )

        unet = UNet3D(
            in_channels=in_channels,
            num_classes=5,
            channels=list(
                self.channels
            ),
            strides=list(
                strides
            ),
            prelu=prelu,
        )

        # Encoder only
        self.conv1 = unet.conv1
        self.conv2 = unet.conv2
        self.conv3 = unet.conv3
        self.conv4 = unet.conv4
        self.conv5 = unet.conv5
        self.bottleneck = unet.bottleneck

        self.pool = nn.AdaptiveAvgPool3d(
            1
        )

        image_feature_dim = (
            self.channels[-1]
        )

        classifier_input_dim = (
            image_feature_dim
            + self.metadata_dim
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                classifier_input_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
        tooth_features=None,
    ):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.bottleneck(x)

        x = self.pool(x)
        x = torch.flatten(
            x,
            1,
        )

        if self.metadata_dim > 0:
            if tooth_features is None:
                raise ValueError(
                    "tooth_features must be provided "
                    "when metadata_dim > 0."
                )

            tooth_features = (
                tooth_features.to(
                    dtype=x.dtype,
                    device=x.device,
                )
            )

            if (
                tooth_features.ndim != 2
                or tooth_features.shape[1]
                != self.metadata_dim
            ):
                raise ValueError(
                    "Expected tooth_features shape "
                    f"(B, {self.metadata_dim}), "
                    f"got {tuple(tooth_features.shape)}."
                )

            x = torch.cat(
                [
                    x,
                    tooth_features,
                ],
                dim=1,
            )

        return self.classifier(x)
