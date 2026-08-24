from __future__ import annotations

import argparse
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

# The shared project temp mount can leave multiprocessing cleanup directories
# busy. Keep worker IPC temporary files on the local node filesystem.
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TMP"] = "/tmp"
tempfile.tempdir = "/tmp"

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from prognosis_dataset_tooth_metadata import PrognosisDataset
from prognosis_model_tooth_metadata import DualBranchPrognosisModel

try:
    import wandb
except ImportError:
    wandb = None


SEED = 42
TARGET_SHAPE = np.array([176, 160, 288])
NUM_CLASSES = 2
CLASS_NAMES = ["Healed", "Not-healed"]
USE_TOOTH_METADATA = True
TOOTH_METADATA_DIM = 4
USE_LOCAL_GLOBAL = True
GLOBAL_DOWNSAMPLE_FACTOR = 2
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
RANKING_BANK_SIZE = 64
RANKING_LOSS_WEIGHT = 0.5
RANKING_MARGIN = 0.2
NUM_WORKERS = 8
NUM_EPOCHS = 200
SCRATCH_LR = 1e-4
PRETRAINED_ENCODER_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4
PRELU = False
GRAD_CLIP_NORM = 5.0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 4
MIN_SCRATCH_LR = 1e-6
MIN_ENCODER_LR = 1e-7
MIN_CLASSIFIER_LR = 1e-6
TTA_PASSES = 5
CHECKPOINT_TAG = "weighted_sampler_auc_v4"

# The stronger augmentation settings used by the notebook's training experiment.
AUGMENTATION = {
    "rotation_degrees": 180.0,
    "translation_voxels": 40.0,
    "spatial_aug_prob": 0.90,
    "affine_prob": 0.50,
    "scale_range": (0.70, 1.30),
    "shear_range": 0.10,
    "elastic_prob": 0.20,
    "elastic_sigma_range": (3.0, 7.0),
    "elastic_magnitude_range": (5.0, 15.0),
    "flip_prob": 0.50,
    "intensity_aug_prob": 0.50,
    "intensity_scale_range": (0.75, 1.25),
    "intensity_shift_fraction": 0.15,
    "noise_prob": 0.40,
    "noise_std_fraction": 0.10,
    "blur_prob": 0.40,
    "blur_sigma_range": (0.5, 1.0),
    "lowres_prob": 0.40,
    "lowres_zoom_range": (0.50, 1.0),
    "gamma_prob": 0.40,
    "gamma_range": (0.70, 1.50),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def normalize_case_id(value) -> str | None:
    match = re.search(r"DSA[-_ ]?0*(\d+)", str(value), re.IGNORECASE)
    return None if match is None else f"DSA{int(match.group(1)):03d}"


def extract_pai(value) -> int | None:
    match = re.search(r"\d+", str(value))
    return None if match is None else int(match.group())


def assign_label(row) -> int | None:
    post_raw = str(row["Follow-up CBCT-PAI [POST]"]).strip()
    if post_raw.lower() in {"-", "", "nan", "none"}:
        return None
    if "extract" in post_raw.lower():
        return 2
    pre = extract_pai(row["Pre-op CBCT-PAI [PRE]"])
    post = extract_pai(row["Follow-up CBCT-PAI [POST]"])
    if post is None:
        return None
    if post <= 2:
        return 0
    if pre is None:
        return None
    return 1 if post < pre else 2


def normalize_column_name(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_tooth_number_column(dataframe: pd.DataFrame) -> str:
    normalized = {normalize_column_name(col): col for col in dataframe.columns}
    for candidate in ("toothnumber", "toothno", "toothnum", "tooth", "toothid", "tooth#"):
        if normalize_column_name(candidate) in normalized:
            return normalized[normalize_column_name(candidate)]
    for column in dataframe.columns:
        key = normalize_column_name(column)
        if "tooth" in key and any(token in key for token in ("number", "num", "no")):
            return column
    raise KeyError("Could not identify the tooth-number column.")


def resolve_full_image_path(case_id: str, root_dir: Path) -> str | None:
    digits = re.findall(r"\d+", str(case_id))
    if not digits:
        return None
    case_number = digits[-1]
    stems = dict.fromkeys([
        str(case_id),
        str(case_id).replace("DSA", "DSA-"),
        f"DSA-{int(case_number):03d}",
        f"DSA{int(case_number):03d}",
        f"DSA-{case_number}",
        f"DSA{case_number}",
    ])
    candidates = []
    for stem in stems:
        candidates.extend([
            root_dir / f"{stem}_img.nii.gz",
            root_dir / f"{stem}pre_img.nii.gz",
            root_dir / f"{stem}PRE_img.nii.gz",
            root_dir / f"{stem}.nii.gz",
            root_dir / f"{stem}_img.nii",
            root_dir / f"{stem}.nii",
        ])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for candidate in sorted(root_dir.glob("*img.nii.gz")):
        if case_number in candidate.name:
            return str(candidate)
    return None


def build_samples(data_dir: Path, outcome_xlsx: Path, global_root: Path):
    discovered = PrognosisDataset(data_dir=data_dir)
    kept = [
        sample for sample in discovered.samples
        if np.all(np.asarray(sample["shape"]) <= TARGET_SHAPE)
    ]
    discovered.samples = kept
    discovered.target_shape = TARGET_SHAPE.copy()

    dataframe = pd.read_excel(outcome_xlsx)
    tooth_column = find_tooth_number_column(dataframe)
    dataframe["normalized_case_id"] = dataframe["Sequence"].apply(normalize_case_id)
    dataframe["label"] = dataframe.apply(assign_label, axis=1)
    dataframe["tooth_number"] = pd.to_numeric(dataframe[tooth_column], errors="coerce")
    dataframe = dataframe[dataframe["normalized_case_id"].notna()]
    dataframe = dataframe[dataframe["label"].notna()].copy()
    if USE_TOOTH_METADATA:
        dataframe = dataframe[dataframe["tooth_number"].between(1, 32)].copy()
    dataframe["label"] = dataframe["label"].astype(int)
    labels = dict(zip(dataframe["normalized_case_id"], dataframe["label"]))
    teeth = dict(zip(dataframe["normalized_case_id"], dataframe["tooth_number"].astype(int)))

    samples = []
    for sample in discovered.samples:
        case_key = normalize_case_id(sample["case_id"])
        if case_key not in labels or (USE_TOOTH_METADATA and case_key not in teeth):
            continue
        item = {
            "case_id": sample["case_id"],
            "image": str(sample["image"]),
            "label": 0 if labels[case_key] == 0 else 1,
        }
        if USE_TOOTH_METADATA:
            item["tooth_number"] = teeth[case_key]
        full_image = resolve_full_image_path(case_key, global_root)
        if USE_LOCAL_GLOBAL:
            if full_image is None:
                continue
            item["full_image"] = full_image
        samples.append(item)
    return samples


def get_train_stats(train_samples, data_dir: Path) -> dict:
    foreground = []
    for sample in tqdm(train_samples, desc="Training intensity statistics"):
        case_id = sample["case_id"]
        refined = data_dir / f"{case_id}_roi_seg_refined.nii.gz"
        original = data_dir / f"{case_id}_roi_seg.nii.gz"
        seg_path = refined if refined.exists() else original
        if not seg_path.exists():
            continue
        image = nib.load(sample["image"]).get_fdata().astype(np.float32)
        segmentation = nib.load(seg_path).get_fdata()
        if image.shape != segmentation.shape:
            raise ValueError(f"Image/segmentation mismatch for {case_id}.")
        values = image[segmentation != 0]
        if values.size:
            foreground.append(values.astype(np.float32, copy=False))
    if not foreground:
        raise RuntimeError("No training foreground voxels were found.")
    values = np.concatenate(foreground)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(np.percentile(values, 0.05)),
        "max": float(np.percentile(values, 99.5)),
    }


def collate_batch(samples):
    """Stack tensors and pad variable-size global volumes within each batch."""
    batch = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if not torch.is_tensor(values[0]):
            batch[key] = values
            continue
        if all(value.shape == values[0].shape for value in values):
            batch[key] = torch.stack([value.contiguous() for value in values])
            continue
        max_shape = tuple(max(value.shape[dim] for value in values) for dim in range(values[0].ndim))
        padded_values = []
        for value in values:
            padded = value.new_zeros(max_shape)
            padded[tuple(slice(0, size) for size in value.shape)] = value
            padded_values.append(padded)
        batch[key] = torch.stack(padded_values)
    return batch


def make_datasets(samples, data_dir, train_stats):
    common = {
        "data_dir": data_dir,
        "train_stats": train_stats,
        "target_shape": TARGET_SHAPE,
        "use_tooth_metadata": USE_TOOTH_METADATA,
        "use_global_branch": USE_LOCAL_GLOBAL,
        "global_downsample_factor": GLOBAL_DOWNSAMPLE_FACTOR,
    }
    train = PrognosisDataset(samples=samples[0], augment=True, **common, **AUGMENTATION)
    val = PrognosisDataset(samples=samples[1], augment=False, **common)
    test = PrognosisDataset(samples=samples[2], augment=False, **common)
    tta_val = PrognosisDataset(samples=samples[1], tta=True, **common)
    tta_test = PrognosisDataset(samples=samples[2], tta=True, **common)
    return train, val, test, tta_val, tta_test


def compute_metrics(y_true, probabilities, predictions=None):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if predictions is None:
        predictions = np.argmax(probabilities, axis=1)
    predictions = np.asarray(predictions, dtype=int)
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_acc": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
    }
    try:
        result["macro_auc"] = roc_auc_score(y_true, probabilities[:, 1])
    except ValueError:
        result["macro_auc"] = np.nan
    recalls = recall_score(y_true, predictions, labels=[0, 1], average=None, zero_division=0)
    result.update({f"recall_class_{i}": float(value) for i, value in enumerate(recalls)})
    return result


CLASS_WEIGHTS = None


def weighted_cross_entropy(logits, labels):
    if CLASS_WEIGHTS is None:
        raise RuntimeError("CLASS_WEIGHTS must be initialized before loss evaluation.")
    return F.cross_entropy(logits, labels, weight=CLASS_WEIGHTS)


def auc_margin_loss(logits, labels, ranking_bank):
    scores = logits[:, 1] - logits[:, 0]
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    bank_positive = ranking_bank["positive"]
    bank_negative = ranking_bank["negative"]
    positive_pool = torch.cat([positives, torch.stack(bank_positive)]) if bank_positive and positives.numel() else (torch.stack(bank_positive) if bank_positive else positives)
    negative_pool = torch.cat([negatives, torch.stack(bank_negative)]) if bank_negative and negatives.numel() else (torch.stack(bank_negative) if bank_negative else negatives)
    terms = []
    if positives.numel() and negative_pool.numel():
        terms.append(F.softplus(RANKING_MARGIN - (positives[:, None] - negative_pool[None, :])).mean())
    if negatives.numel() and positive_pool.numel():
        terms.append(F.softplus(RANKING_MARGIN - (positive_pool[:, None] - negatives[None, :])).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def training_loss(logits, labels, ranking_bank):
    return weighted_cross_entropy(logits, labels) + RANKING_LOSS_WEIGHT * auc_margin_loss(logits, labels, ranking_bank)


def update_ranking_bank(ranking_bank, logits, labels):
    scores = (logits[:, 1] - logits[:, 0]).detach()
    ranking_bank["positive"].extend(scores[labels == 1].unbind())
    ranking_bank["negative"].extend(scores[labels == 0].unbind())
    for key in ("positive", "negative"):
        ranking_bank[key] = ranking_bank[key][-RANKING_BANK_SIZE:]


def move_batch(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    global_images = batch.get("global_image")
    if global_images is not None:
        global_images = global_images.to(device, non_blocking=True)
    labels = batch.get("label")
    if labels is not None:
        labels = labels.to(device, non_blocking=True)
    tooth_features = batch.get("tooth_features")
    if tooth_features is not None:
        tooth_features = tooth_features.to(device, non_blocking=True)
    return images, global_images, labels, tooth_features


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, ranking_bank):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    seen = 0
    y_true, y_prob = [], []
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False), 1):
        images, global_images, labels, tooth_features = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images, global_images, tooth_features)
            raw_loss = training_loss(logits, labels, ranking_bank)
            loss = raw_loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        update_ranking_bank(ranking_bank, logits, labels)
        if step % GRAD_ACCUM_STEPS == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = images.size(0)
        seen += batch_size
        total_loss += raw_loss.item() * batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_prob.extend(torch.softmax(logits.detach().float(), dim=1).cpu().numpy().tolist())
    return total_loss / seen, compute_metrics(y_true, y_prob)


@torch.no_grad()
def evaluate_once(model, loader, device):
    model.eval()
    total_loss = 0.0
    seen = 0
    y_true, y_prob, case_ids = [], [], []
    for batch in loader:
        images, global_images, labels, tooth_features = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images, global_images, tooth_features)
            loss = weighted_cross_entropy(logits, labels)
        batch_size = images.size(0)
        seen += batch_size
        total_loss += loss.item() * batch_size
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(torch.softmax(logits.float(), dim=1).cpu().numpy().tolist())
        case_ids.extend(batch["case_id"])
    probabilities = np.asarray(y_prob)
    return {
        "loss": total_loss / seen,
        "metrics": compute_metrics(y_true, probabilities),
        "y_true": np.asarray(y_true),
        "y_prob": probabilities,
        "y_pred": np.argmax(probabilities, axis=1),
        "case_ids": case_ids,
    }


def calibrate_threshold(y_true, probabilities):
    """Choose the probability threshold maximizing Youden's J statistic."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(y_true).size < 2:
        raise ValueError("Youden's J threshold calibration requires both classes in the training set.")
    fpr, tpr, thresholds = roc_curve(y_true, probabilities[:, 1])
    j_scores = tpr - fpr
    best_indices = np.flatnonzero(j_scores == np.max(j_scores))
    # Prefer the threshold closest to 0.5 when several thresholds have the same J.
    best_index = best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))]
    return float(thresholds[best_index]), float(j_scores[best_index])


def evaluate_tta(model, dataset, loader_kwargs, device, passes):
    results = []
    for pass_index in range(passes):
        loader = DataLoader(dataset, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
        result = evaluate_once(model, loader, device)
        results.append(result)
        print(f"TTA pass {pass_index + 1}/{passes}: AUC={result['metrics']['macro_auc']:.4f}")
    probabilities = np.mean([result["y_prob"] for result in results], axis=0)
    output = dict(results[0])
    output["y_prob"] = probabilities
    output["y_pred"] = np.argmax(probabilities, axis=1)
    output["metrics"] = compute_metrics(output["y_true"], probabilities)
    output["loss"] = float(np.mean([result["loss"] for result in results]))
    return output


def evaluate_checkpoint_variants(
    model,
    checkpoint_path,
    checkpoint_name,
    test_ds,
    tta_test_ds,
    loader_kwargs,
    device,
    tta_passes,
    output_dir,
    decision_threshold,
    run=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = DataLoader(test_ds, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
    results = {
        "no_tta": evaluate_once(model, test_loader, device),
        "tta": evaluate_tta(model, tta_test_ds, loader_kwargs, device, tta_passes),
    }
    for variant, result in results.items():
        result["y_pred"] = (result["y_prob"][:, 1] >= decision_threshold).astype(int)
        result["metrics"] = compute_metrics(result["y_true"], result["y_prob"], result["y_pred"])
        metrics = result["metrics"]
        print(
            f"{checkpoint_name} | {variant} | AUC: {metrics['macro_auc']:.4f} | "
            f"Balanced accuracy: {metrics['balanced_acc']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )
        predictions = pd.DataFrame({
            "case_id": result["case_ids"],
            "label": result["y_true"],
            "prediction": result["y_pred"],
            "prob_healed": result["y_prob"][:, 0],
            "prob_not_healed": result["y_prob"][:, 1],
        })
        predictions.to_csv(output_dir / f"test_predictions_{checkpoint_name}_{variant}.csv", index=False)
        if run:
            run.log({
                f"test/{checkpoint_name}/{variant}/loss": result["loss"],
                **{f"test/{checkpoint_name}/{variant}/{key}": value for key, value in metrics.items()},
                f"test/{checkpoint_name}/{variant}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=result["y_true"].tolist(),
                    preds=result["y_pred"].tolist(),
                    class_names=CLASS_NAMES,
                ),
            })
    return results


def load_pretrained_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "unet"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    cleaned = {}
    for key, value in checkpoint.items():
        clean_key = key
        for prefix in ("module.", "model.", "unet."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    translations = {
        "encoders.0.": "conv1.", "encoders.1.": "conv2.", "encoders.2.": "conv3.",
        "encoders.3.": "conv4.", "encoders.4.": "conv5.", "encoders.5.": "bottleneck.",
    }
    for target in (model.local_encoder, model.global_encoder):
        target_state = target.state_dict()
        valid = {}
        for key, value in cleaned.items():
            mapped = key
            for old, new in translations.items():
                if key.startswith(old):
                    mapped = new + key[len(old):]
                    break
            if mapped in target_state and target_state[mapped].shape == value.shape:
                valid[mapped] = value
            elif key in target_state and target_state[key].shape == value.shape:
                valid[key] = value
        if not valid:
            raise RuntimeError("No compatible pretrained encoder tensors found.")
        result = target.load_state_dict(valid, strict=False)
        print(f"Loaded {len(valid)} encoder tensors; missing={len(result.missing_keys)}")


def main():
    parser = argparse.ArgumentParser(description="Train local/global CBCT prognosis model for AUC.")
    parser.add_argument("--data-dir", type=Path, default=Path("../DSApre/roi_crop"))
    parser.add_argument("--global-root", type=Path, default=Path("../DSApre"))
    parser.add_argument("--outcome-xlsx", type=Path, default=Path("Dataset A - Overview.xlsx"))
    parser.add_argument("--pretrained", type=Path, default=Path("epoch900_Proposed_small_lesion30_zeroshot_small_lr_ulb80.pth"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--tta-passes", type=int, default=TTA_PASSES)
    parser.add_argument("--wandb-project", default="dental-prognosis")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0 or args.tta_passes < 1:
        raise ValueError("batch size, TTA passes, and worker count must be valid positive values.")
    if args.wandb_mode != "disabled" and wandb is None:
        raise RuntimeError("Install wandb or pass --wandb-mode disabled.")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args.data_dir, args.outcome_xlsx, args.global_root)
    train_data, temp_data = train_test_split(samples, test_size=0.30, random_state=SEED, stratify=[x["label"] for x in samples])
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=SEED, stratify=[x["label"] for x in temp_data])
    split_rows = [{"split": name, **sample} for name, data in (("train", train_data), ("val", val_data), ("test", test_data)) for sample in data]
    pd.DataFrame(split_rows).to_csv(args.checkpoint_dir / f"prelim_split_seed42_{CHECKPOINT_TAG}.csv", index=False)
    train_stats = get_train_stats(train_data, args.data_dir)
    train_ds, val_ds, test_ds, tta_val_ds, tta_test_ds = make_datasets((train_data, val_data, test_data), args.data_dir, train_stats)
    # Calibrate on the original training examples, without augmentation.
    train_calibration_ds = PrognosisDataset(
        samples=train_data,
        augment=False,
        data_dir=args.data_dir,
        train_stats=train_stats,
        target_shape=TARGET_SHAPE,
        use_tooth_metadata=USE_TOOTH_METADATA,
        use_global_branch=USE_LOCAL_GLOBAL,
        global_downsample_factor=GLOBAL_DOWNSAMPLE_FACTOR,
    )

    train_labels = np.asarray([x["label"] for x in train_data])
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    sampler_weights = np.where(train_labels == 1, 0.5 / class_counts[1], 0.5 / class_counts[0])
    global CLASS_WEIGHTS
    CLASS_WEIGHTS = torch.as_tensor(
        len(train_labels) / (NUM_CLASSES * class_counts),
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(torch.as_tensor(sampler_weights, dtype=torch.double), len(train_ds), replacement=True, generator=generator)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_ds, sampler=sampler, generator=generator, collate_fn=collate_batch, **loader_kwargs)

    model = DualBranchPrognosisModel(in_channels=1, num_classes=NUM_CLASSES, dropout=DROPOUT, metadata_dim=TOOTH_METADATA_DIM if USE_TOOTH_METADATA else 0, prelu=PRELU, use_global_branch=USE_LOCAL_GLOBAL).to(device)
    use_pretrained = args.pretrained is not None and args.pretrained.exists()
    if use_pretrained:
        load_pretrained_encoder(model, args.pretrained)
    encoder_params, classifier_params = [], []
    for name, parameter in model.named_parameters():
        (classifier_params if any(k in name for k in ("classifier", "head", "fc")) else encoder_params).append(parameter)
    groups = ([{"params": encoder_params, "lr": PRETRAINED_ENCODER_LR, "name": "encoder"}, {"params": classifier_params, "lr": CLASSIFIER_LR, "name": "classifier"}] if use_pretrained else [{"params": model.parameters(), "lr": SCRATCH_LR, "name": "all"}])
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[PRETRAINED_ENCODER_LR, CLASSIFIER_LR] if use_pretrained else SCRATCH_LR,
        epochs=args.epochs,
        steps_per_epoch=updates_per_epoch,
        pct_start=0.10,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = {
        "seed": SEED, "device": str(device), "training_mode": "pretrained" if use_pretrained else "scratch",
        "pretrained_path": str(args.pretrained) if use_pretrained else None, "batch_size": args.batch_size,
        "grad_accum_steps": GRAD_ACCUM_STEPS, "effective_batch_size": args.batch_size * GRAD_ACCUM_STEPS,
        "num_workers": args.num_workers, "num_epochs": args.epochs, "tta_passes": args.tta_passes,
        "target_shape": TARGET_SHAPE.tolist(), "use_local_global": USE_LOCAL_GLOBAL,
        "global_downsample_factor": GLOBAL_DOWNSAMPLE_FACTOR, "use_tooth_metadata": USE_TOOTH_METADATA,
        "tooth_metadata_dim": TOOTH_METADATA_DIM, "class_names": CLASS_NAMES, "dropout": DROPOUT,
        "prelu": PRELU, "ranking_loss_weight": RANKING_LOSS_WEIGHT,
        "ranking_margin": RANKING_MARGIN, "ranking_bank_size": RANKING_BANK_SIZE, "scratch_lr": SCRATCH_LR,
        "pretrained_encoder_lr": PRETRAINED_ENCODER_LR, "classifier_lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM, "class_counts_train": class_counts.tolist(),
        "sampler_weights": sampler_weights.tolist(), "class_weights_loss": CLASS_WEIGHTS.cpu().tolist(), "classification_loss": "weighted_cross_entropy", "intensity_stats": train_stats, "augmentation": AUGMENTATION,
        "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds),
    }
    run = None if args.wandb_mode == "disabled" else wandb.init(project=args.wandb_project, name=f"{'pretrained' if use_pretrained else 'scratch'}-{CHECKPOINT_TAG}", mode=args.wandb_mode, config=config)

    ranking_bank = {"positive": [], "negative": []}
    best_val_loss = np.inf
    best_epoch = -1
    run_prefix = f"{'pretrained' if use_pretrained else 'scratch'}_{CHECKPOINT_TAG}"
    best_path = args.checkpoint_dir / f"best_{run_prefix}.pth"
    final_path = args.checkpoint_dir / f"final_{run_prefix}.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, ranking_bank)
        val_loader = DataLoader(val_ds, shuffle=False, collate_fn=collate_batch, **loader_kwargs)
        val_result = evaluate_once(model, val_loader, device)
        val_metrics = val_result["metrics"]
        val_auc = val_metrics["macro_auc"]
        learning_rates = {"lr/encoder": optimizer.param_groups[0]["lr"], "lr/classifier": optimizer.param_groups[1]["lr"] if use_pretrained else optimizer.param_groups[0]["lr"]}
        print(f"Epoch {epoch:03d} | train loss={train_loss:.4f} | val loss={val_result['loss']:.4f} | val AUC={val_auc:.4f}")
        if run:
            run.log({"epoch": epoch, "train/loss": train_loss, **{f"train/{k}": v for k, v in train_metrics.items()}, "val/loss": val_result["loss"], **{f"val/{k}": v for k, v in val_metrics.items()}, **learning_rates}, step=epoch)
        val_loss = val_result["loss"]
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_loss": best_val_loss, "best_val_auc": val_auc, "train_stats": train_stats, "target_shape": TARGET_SHAPE, "class_names": CLASS_NAMES}, best_path)

    # Use the final model's training-set probabilities to calibrate the decision threshold.
    train_calibration_loader = DataLoader(
        train_calibration_ds,
        shuffle=False,
        collate_fn=collate_batch,
        **loader_kwargs,
    )
    train_calibration_result = evaluate_once(model, train_calibration_loader, device)
    decision_threshold, youden_j = calibrate_threshold(
        train_calibration_result["y_true"], train_calibration_result["y_prob"]
    )
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "train_stats": train_stats,
        "target_shape": TARGET_SHAPE,
        "class_names": CLASS_NAMES,
        "decision_threshold": decision_threshold,
        "youden_j": youden_j,
    }, final_path)
    print(f"Final-model training threshold: {decision_threshold:.6f} (Youden's J={youden_j:.6f})")

    final_checkpoint_name = f"final_{run_prefix}"
    evaluate_checkpoint_variants(
        model=model,
        checkpoint_path=final_path,
        checkpoint_name=final_checkpoint_name,
        test_ds=test_ds,
        tta_test_ds=tta_test_ds,
        loader_kwargs=loader_kwargs,
        device=device,
        tta_passes=args.tta_passes,
        output_dir=args.checkpoint_dir,
        decision_threshold=decision_threshold,
        run=run,
    )
    if run:
        run.summary.update({
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "decision_threshold": decision_threshold,
            "youden_j": youden_j,
            "final_checkpoint": str(final_path),
        })
        run.finish()


if __name__ == "__main__":
    main()
