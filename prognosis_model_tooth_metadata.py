import sys

import torch
import torch.nn as nn

# sys.path.append("../Segmentation")

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


class DualBranchPrognosisModel(nn.Module):
    """
    Local-global prognosis model.

    The cropped ROI and the downsampled full-volume image are each encoded by an
    identical UNet3D encoder. The same pretrained encoder weights are loaded into
    both branches, but each branch remains independently trainable.
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
        use_global_branch=True,
    ):
        super().__init__()

        self.use_global_branch = bool(use_global_branch)
        self.channels = tuple(int(x) for x in channels)
        self.metadata_dim = int(metadata_dim)
        self.input_dim = int(in_channels)

        def make_unet():
            return UNet3D(
                in_channels=in_channels,
                num_classes=5,
                channels=list(self.channels),
                strides=list(strides),
                prelu=prelu,
            )

        self.local_encoder = make_unet()
        self.global_encoder = make_unet()

        self.pool = nn.AdaptiveAvgPool3d(1)
        image_feature_dim = self.channels[-1]

        fusion_dim = 2 * image_feature_dim + self.metadata_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    @staticmethod
    def _encoder_feature_vector(encoder, x):
        x = encoder.conv1(x)
        x = encoder.conv2(x)
        x = encoder.conv3(x)
        x = encoder.conv4(x)
        x = encoder.conv5(x)
        x = encoder.bottleneck(x)

        x = nn.functional.adaptive_avg_pool3d(x, 1)
        return torch.flatten(x, 1)

    def load_shared_pretrained_encoder(self, state_dict):
        """
        Load the same encoder checkpoint into both branches.
        """
        encoder_prefixes = (
            "conv1.",
            "conv2.",
            "conv3.",
            "conv4.",
            "conv5.",
            "bottleneck.",
        )

        shared = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(encoder_prefixes)
        }

        if len(shared) == 0:
            raise RuntimeError(
                "No encoder tensors matched the dual-branch model. "
                "Check the checkpoint key names."
            )

        self.local_encoder.load_state_dict(shared, strict=False)
        self.global_encoder.load_state_dict(shared, strict=False)

    def forward(self, local_x, global_x=None, tooth_features=None):
        local_feat = self._encoder_feature_vector(
            self.local_encoder,
            local_x,
        )

        if self.use_global_branch:
            if global_x is None:
                raise ValueError(
                    "global_x must be provided when "
                    "use_global_branch=True."
                )
            global_feat = self._encoder_feature_vector(
                self.global_encoder,
                global_x,
            )
        else:
            global_feat = torch.zeros_like(local_feat)

        fused = torch.cat([local_feat, global_feat], dim=1)

        if self.metadata_dim > 0:
            if tooth_features is None:
                raise ValueError(
                    "tooth_features must be provided when metadata_dim > 0."
                )

            tooth_features = tooth_features.to(
                dtype=fused.dtype,
                device=fused.device,
            )

            if tooth_features.ndim != 2 or tooth_features.shape[1] != self.metadata_dim:
                raise ValueError(
                    "Expected tooth_features shape "
                    f"(B, {self.metadata_dim}), got {tuple(tooth_features.shape)}."
                )

            fused = torch.cat([fused, tooth_features], dim=1)

        return self.classifier(fused)
