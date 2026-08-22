from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset


class PrognosisDataset(Dataset):
    """
    Dataset for 3D CBCT prognosis classification.

    Two usage modes
    ----------------
    1) Discovery mode:
       PrognosisDataset(data_dir=...)
       -> scans available ROI images and builds self.samples.

    2) Training/evaluation mode:
       PrognosisDataset(
           data_dir=...,
           samples=train_data,
           train_stats=train_stats,
           target_shape=(176, 160, 288),
           augment=True,   # train only
       )
       -> returns augmented/normalized/padded image and integer label.

    Augmentations are intentionally conservative because lesion size,
    cortical involvement, and surrounding anatomy can be prognostic.
    Therefore, random resizing/cropping and arbitrary flips are not used.
    """

    def __init__(
        self,
        data_dir,
        samples=None,
        train_stats=None,
        target_shape=None,
        augment=False,
        tta=False,
        use_tooth_metadata=False,
        use_global_branch=False,
        global_downsample_factor=1,
        rotation_degrees=10.0,
        translation_voxels=6.0,
        spatial_aug_prob=0.5,
        intensity_aug_prob=0.5,
        intensity_scale_range=(0.90, 1.10),
        intensity_shift_fraction=0.05,
        noise_prob=0.30,
        noise_std_fraction=0.01,
        affine_prob=0.50,
        scale_range=(0.70, 1.30),
        shear_range=0.10,
        elastic_prob=0.20,
        elastic_sigma_range=(3.0, 7.0),
        elastic_magnitude_range=(5.0, 15.0),
        flip_prob=0.50,
        blur_prob=0.40,
        blur_sigma_range=(0.5, 1.0),
        lowres_prob=0.40,
        lowres_zoom_range=(0.50, 1.0),
        gamma_prob=0.40,
        gamma_range=(0.70, 1.50),
    ):
        self.data_dir = Path(data_dir)
        self.train_stats = train_stats

        self.augment = bool(augment)
        self.tta = bool(tta)
        self.use_tooth_metadata = bool(use_tooth_metadata)
        self.use_global_branch = bool(use_global_branch)
        self.global_downsample_factor = int(global_downsample_factor)
        if self.global_downsample_factor < 1:
            raise ValueError("global_downsample_factor must be >= 1.")
        self.rotation_degrees = float(rotation_degrees)
        self.translation_voxels = float(translation_voxels)
        self.spatial_aug_prob = float(spatial_aug_prob)
        self.intensity_aug_prob = float(intensity_aug_prob)
        self.intensity_scale_range = tuple(intensity_scale_range)
        self.intensity_shift_fraction = float(intensity_shift_fraction)
        self.noise_prob = float(noise_prob)
        self.noise_std_fraction = float(noise_std_fraction)
        self.affine_prob = float(affine_prob)
        self.scale_range = tuple(scale_range)
        self.shear_range = float(shear_range)
        self.elastic_prob = float(elastic_prob)
        self.elastic_sigma_range = tuple(elastic_sigma_range)
        self.elastic_magnitude_range = tuple(elastic_magnitude_range)
        self.flip_prob = float(flip_prob)
        self.blur_prob = float(blur_prob)
        self.blur_sigma_range = tuple(blur_sigma_range)
        self.lowres_prob = float(lowres_prob)
        self.lowres_zoom_range = tuple(lowres_zoom_range)
        self.gamma_prob = float(gamma_prob)
        self.gamma_range = tuple(gamma_range)

        if samples is None:
            self.samples = []

            for original_path in sorted(
                self.data_dir.glob("*_roi_img.nii.gz")
            ):
                refined_path = original_path.with_name(
                    original_path.name.replace(
                        "_roi_img.nii.gz",
                        "_roi_img_refined.nii.gz",
                    )
                )

                img_path = (
                    refined_path
                    if refined_path.exists()
                    else original_path
                )

                case_id = original_path.name.replace(
                    "_roi_img.nii.gz", ""
                )

                nii = nib.load(img_path)

                self.samples.append(
                    {
                        "case_id": case_id,
                        "image": str(img_path),
                        "shape": np.array(nii.shape[:3]),
                    }
                )

        else:
            self.samples = []

            for sample in samples:
                s = dict(sample)

                img_path = Path(s["image"])
                s["image"] = str(img_path)

                if "case_id" not in s:
                    name = img_path.name
                    name = name.replace(
                        "_roi_img_refined.nii.gz", ""
                    )
                    name = name.replace(
                        "_roi_img.nii.gz", ""
                    )
                    s["case_id"] = name

                if "label" not in s and "class" in s:
                    s["label"] = int(s["class"])

                if "shape" not in s:
                    s["shape"] = np.array(
                        nib.load(img_path).shape[:3]
                    )

                self.samples.append(s)

        if len(self.samples) == 0:
            raise ValueError("No samples were found.")

        shapes = np.stack(
            [
                np.asarray(sample["shape"], dtype=int)
                for sample in self.samples
            ]
        )

        if target_shape is None:
            self.target_shape = shapes.max(axis=0)
        else:
            self.target_shape = np.asarray(
                target_shape,
                dtype=int,
            )

        too_large = np.any(
            shapes > self.target_shape,
            axis=1,
        )
        if np.any(too_large):
            bad_ids = [
                self.samples[i]["case_id"]
                for i in np.where(too_large)[0]
            ]
            raise ValueError(
                "Some images are larger than target_shape: "
                + ", ".join(bad_ids)
            )

        if self.augment and self.train_stats is None:
            raise ValueError(
                "train_stats must be provided when augment=True."
            )

    def __len__(self):
        return len(self.samples)

    def tta_image(self, img):
        """Apply a light stochastic transform for test-time augmentation."""
        img = img.astype(np.float32, copy=False)
        a_min = float(self.train_stats["min"])

        if np.random.rand() < 0.5:
            img = np.flip(img, axis=np.random.randint(0, 3))

        angle = np.random.uniform(-5.0, 5.0)
        plane = ((0, 1), (0, 2), (1, 2))[np.random.randint(0, 3)]
        return ndi.rotate(
            img,
            angle=angle,
            axes=plane,
            reshape=False,
            order=1,
            mode="constant",
            cval=a_min,
            prefilter=False,
        )

    def augment_image(self, img):
        """
        Conservative train-only 3D augmentation.

        Spatial:
        - random rotation and translation
        - random scale/shear affine transform
        - elastic deformation
        - random axis flip

        Intensity:
        - contrast/intensity scaling and shift
        - Gaussian noise and blur
        - low-resolution simulation and gamma adjustment
        """
        img = img.astype(
            np.float32,
            copy=False,
        )
        original_shape = img.shape

        a_min = float(
            self.train_stats["min"]
        )
        a_max = float(
            self.train_stats["max"]
        )
        intensity_range = (
            a_max - a_min
        )

        if np.random.rand() < self.spatial_aug_prob:
            plane = (
                (0, 1),
                (0, 2),
                (1, 2),
            )[np.random.randint(0, 3)]

            angle = np.random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            img = ndi.rotate(
                img,
                angle=angle,
                axes=plane,
                reshape=False,
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            )

        if np.random.rand() < self.spatial_aug_prob:
            shift = np.random.uniform(
                -self.translation_voxels,
                self.translation_voxels,
                size=3,
            )

            img = ndi.shift(
                img,
                shift=shift,
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            )

        if np.random.rand() < self.affine_prob:
            scale = np.random.uniform(
                self.scale_range[0],
                self.scale_range[1],
                size=3,
            )
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range,
                size=(3, 3),
            )
            np.fill_diagonal(shear, 0.0)
            matrix = np.diag(scale) @ (np.eye(3) + shear)
            center = (np.asarray(img.shape) - 1.0) / 2.0
            offset = center - matrix @ center
            img = ndi.affine_transform(
                img,
                matrix=matrix,
                offset=offset,
                output_shape=img.shape,
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            )

        if np.random.rand() < self.elastic_prob:
            sigma = np.random.uniform(
                self.elastic_sigma_range[0],
                self.elastic_sigma_range[1],
            )
            magnitude = np.random.uniform(
                self.elastic_magnitude_range[0],
                self.elastic_magnitude_range[1],
            )
            shape = img.shape
            coordinates = np.meshgrid(
                *[np.arange(size, dtype=np.float32) for size in shape],
                indexing="ij",
            )
            for axis in range(3):
                displacement = ndi.gaussian_filter(
                    np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32),
                    sigma=sigma,
                    mode="nearest",
                )
                displacement *= magnitude / max(
                    float(np.std(displacement)),
                    1e-6,
                )
                coordinates[axis] += displacement
            img = ndi.map_coordinates(
                img,
                coordinates,
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            ).reshape(shape)

        if np.random.rand() < self.flip_prob:
            img = np.flip(
                img,
                axis=np.random.randint(0, 3),
            )

        if np.random.rand() < self.intensity_aug_prob:
            scale = np.random.uniform(
                self.intensity_scale_range[0],
                self.intensity_scale_range[1],
            )

            shift = np.random.uniform(
                -self.intensity_shift_fraction,
                self.intensity_shift_fraction,
            ) * intensity_range

            center = 0.5 * (
                a_min + a_max
            )

            img = (
                (img - center)
                * scale
                + center
                + shift
            )

        if np.random.rand() < self.noise_prob:
            noise_std = (
                self.noise_std_fraction
                * intensity_range
            )

            img = img + np.random.normal(
                loc=0.0,
                scale=noise_std,
                size=img.shape,
            ).astype(np.float32)

        if np.random.rand() < self.blur_prob:
            sigma = np.random.uniform(
                self.blur_sigma_range[0],
                self.blur_sigma_range[1],
            )
            img = ndi.gaussian_filter(
                img,
                sigma=(sigma, sigma, sigma),
            )

        if np.random.rand() < self.lowres_prob:
            zoom = np.random.uniform(
                self.lowres_zoom_range[0],
                self.lowres_zoom_range[1],
            )
            lowres = ndi.zoom(
                img,
                zoom=zoom,
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            )
            img = ndi.zoom(
                lowres,
                zoom=np.asarray(original_shape) / np.asarray(lowres.shape),
                order=1,
                mode="constant",
                cval=a_min,
                prefilter=False,
            )
            if img.shape != original_shape:
                fitted = np.full(
                    original_shape,
                    a_min,
                    dtype=np.float32,
                )
                common_shape = tuple(
                    min(current, target)
                    for current, target in zip(
                        img.shape,
                        original_shape,
                    )
                )
                source_slices = tuple(
                    slice(0, size)
                    for size in common_shape
                )
                fitted[source_slices] = img[source_slices]
                img = fitted
            img = img[
                tuple(slice(0, size) for size in original_shape)
            ]

        if np.random.rand() < self.gamma_prob:
            gamma = np.random.uniform(
                self.gamma_range[0],
                self.gamma_range[1],
            )
            img = np.power(
                np.clip((img - a_min) / intensity_range, 0.0, 1.0),
                gamma,
            ) * intensity_range + a_min

        return img.astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def tooth_metadata_from_number(tooth_number):
        """
        Convert US/Universal tooth number (1-32) into four compact
        anatomical predictors:

        [mandibular, anterior, premolar, molar]

        Maxillary is represented by mandibular=0.
        Tooth type is one-hot encoded.
        """
        tooth_number = int(tooth_number)

        if not 1 <= tooth_number <= 32:
            raise ValueError(
                f"Invalid US/Universal tooth number: {tooth_number}"
            )

        mandibular = float(
            tooth_number >= 17
        )

        anterior_teeth = {
            6, 7, 8, 9, 10, 11,
            22, 23, 24, 25, 26, 27,
        }

        premolar_teeth = {
            4, 5, 12, 13,
            20, 21, 28, 29,
        }

        molar_teeth = {
            1, 2, 3,
            14, 15, 16,
            17, 18, 19,
            30, 31, 32,
        }

        anterior = float(
            tooth_number in anterior_teeth
        )

        premolar = float(
            tooth_number in premolar_teeth
        )

        molar = float(
            tooth_number in molar_teeth
        )

        if (
            anterior
            + premolar
            + molar
        ) != 1:
            raise ValueError(
                f"Could not determine tooth type: {tooth_number}"
            )

        features = np.array(
            [
                mandibular,
                anterior,
                premolar,
                molar,
            ],
            dtype=np.float32,
        )

        arch_name = (
            "Mandibular"
            if mandibular == 1.0
            else "Maxillary"
        )

        if anterior == 1.0:
            tooth_type = "Anterior"
        elif premolar == 1.0:
            tooth_type = "Premolar"
        else:
            tooth_type = "Molar"

        return (
            features,
            arch_name,
            tooth_type,
        )

    @staticmethod
    def normalize_intensity(img, train_stats):
        """
        Match the source segmentation preprocessing:
        1) percentile clipping
        2) scale to [0, 1]
        3) normalize using training-set foreground mean/std
        """
        if train_stats is None:
            raise ValueError(
                "train_stats must be provided for __getitem__."
            )

        img = img.astype(np.float32)

        a_min = float(train_stats["min"])
        a_max = float(train_stats["max"])
        mean = float(train_stats["mean"])
        std = float(train_stats["std"])

        intensity_range = a_max - a_min

        if intensity_range <= 0:
            raise ValueError(
                "train_stats['max'] must be > train_stats['min']."
            )
        if std <= 0:
            raise ValueError(
                "train_stats['std'] must be > 0."
            )

        img = np.clip(
            img,
            a_min,
            a_max,
        )

        img = (
            img - a_min
        ) / intensity_range

        scaled_mean = (
            mean - a_min
        ) / intensity_range

        scaled_std = (
            std / intensity_range
        )

        img = (
            img - scaled_mean
        ) / scaled_std

        return img

    @staticmethod
    def normalized_padding_value(train_stats):
        """
        In source segmentation training, padding value 0 was added
        after scaling to [0,1] and before normalization.
        This returns that same value in normalized intensity space.
        """
        a_min = float(train_stats["min"])
        a_max = float(train_stats["max"])
        mean = float(train_stats["mean"])
        std = float(train_stats["std"])

        intensity_range = a_max - a_min
        scaled_mean = (
            mean - a_min
        ) / intensity_range
        scaled_std = (
            std / intensity_range
        )

        return float(
            (0.0 - scaled_mean)
            / scaled_std
        )

    @staticmethod
    def pad_to_shape(
        img,
        target_shape,
        constant_value,
    ):
        current_shape = np.asarray(
            img.shape,
            dtype=int,
        )
        target_shape = np.asarray(
            target_shape,
            dtype=int,
        )

        diff = (
            target_shape
            - current_shape
        )

        if np.any(diff < 0):
            raise ValueError(
                f"Image shape {tuple(current_shape)} "
                f"is larger than target shape {tuple(target_shape)}."
            )

        before = diff // 2
        after = diff - before

        pad_width = [
            (int(b), int(a))
            for b, a in zip(
                before,
                after,
            )
        ]

        return np.pad(
            img,
            pad_width,
            mode="constant",
            constant_values=constant_value,
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        nii = nib.load(
            sample["image"]
        )

        img = nii.get_fdata().astype(
            np.float32
        )

        original_shape = tuple(
            img.shape[:3]
        )

        if self.augment:
            img = self.augment_image(
                img
            )
        elif self.tta:
            img = self.tta_image(img)

        img = self.normalize_intensity(
            img,
            self.train_stats,
        )

        pad_value = self.normalized_padding_value(
            self.train_stats
        )

        img = self.pad_to_shape(
            img,
            self.target_shape,
            constant_value=pad_value,
        )

        img = torch.tensor(
            img,
            dtype=torch.float32,
        ).unsqueeze(0)

        output = {
            "image": img,
            "case_id": sample["case_id"],
            "original_shape": original_shape,
            "path": str(sample["image"]),
        }

        if self.use_global_branch and "full_image" in sample:
            full_img = nib.load(sample["full_image"]).get_fdata().astype(np.float32)

            if self.global_downsample_factor > 1:
                zoom = [
                    1.0 / self.global_downsample_factor,
                    1.0 / self.global_downsample_factor,
                    1.0 / self.global_downsample_factor,
                ]
                full_img = ndi.zoom(
                    full_img,
                    zoom=zoom,
                    order=1,
                    mode="constant",
                    prefilter=False,
                )

            full_img = self.normalize_intensity(full_img, self.train_stats)
            output["global_image"] = torch.tensor(
                full_img,
                dtype=torch.float32,
            ).unsqueeze(0)

        if self.use_tooth_metadata:
            if "tooth_number" not in sample:
                raise ValueError(
                    f"Missing tooth_number for {sample['case_id']}."
                )

            tooth_features, arch_name, tooth_type = (
                self.tooth_metadata_from_number(
                    sample["tooth_number"]
                )
            )

            output["tooth_number"] = torch.tensor(
                int(sample["tooth_number"]),
                dtype=torch.long,
            )

            output["tooth_features"] = torch.tensor(
                tooth_features,
                dtype=torch.float32,
            )

            output["arch"] = arch_name
            output["tooth_type"] = tooth_type

        if "label" in sample:
            output["label"] = torch.tensor(
                int(sample["label"]),
                dtype=torch.long,
            )

        if "annotation_targets" in sample:
            output["annotation_targets"] = torch.tensor(
                np.asarray(sample["annotation_targets"], dtype=np.int64),
                dtype=torch.long,
            )
            output["annotation_mask"] = torch.tensor(
                np.asarray(
                    sample.get(
                        "annotation_mask",
                        np.ones_like(sample["annotation_targets"]),
                    ),
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

        return output
