import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold scope helpers
# ---------------------------------------------------------------------------

def collect_pixel_values(tiff_paths: list[Path]) -> np.ndarray:
    """Load all tiff files and return a flat array of all pixel values."""
    arrays = []
    for p in tiff_paths:
        arr = tifffile.imread(str(p)).astype(np.float32)
        arrays.append(arr.ravel())
    return np.concatenate(arrays)


def compute_group_threshold(tiff_paths: list[Path]) -> float:
    """Compute threshold = mean + N*std over all pixels in the group."""
    pixels = collect_pixel_values(tiff_paths)
    mean = float(np.mean(pixels))
    std  = float(np.std(pixels))

    threshold = None
    for n, label in enumerate(["only mean", "mean+1std", "mean+2std",
                                "mean+3std", "mean+4std", "mean+5std", "mean+6std", "mean+7std", "mean+8std", "mean+9std", "mean+10std", "mean+11std", "mean+12std",
                                "mean+13std", "mean+14std", "mean+15std", "mean+16std", "mean+17std", "mean+18std", "mean+19std", "mean+20std", "mean+21std", "mean+22std",
                                "mean+23std", "mean+24std", "mean+25std", "mean+26std", "mean+27std", "mean+28std", "mean+29std", "mean+30std"]):
        candidate = mean + n * std
        if candidate > 0.5:
            threshold = candidate
            print(label)
            break

    if threshold is None:
        print("no threshold")
        threshold = mean + 30 * std   

    logger.info(
        f"  Group threshold: mean={mean:.6f}, "
        f"std={std:.6f}, threshold={threshold:.6f}"
    )
    return threshold


def compute_image_threshold(arr: np.ndarray) -> float:
    """Compute threshold = mean + 1.5*std for a single image."""
    pixels = arr.astype(np.float32).ravel()
    return float(np.mean(pixels) + 1.5 * np.std(pixels))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold: >threshold → 255, else → 0. Returns uint8 array."""
    return np.where(arr > threshold, 255, 0).astype(np.uint8)


def process_group_with_threshold(
    tiff_paths: list[Path],
    src_root: Path,
    dst_root: Path,
    threshold: float,
) -> None:
    """Save PNGs using a pre-computed shared threshold."""
    out_path = None
    for tiff_path in tqdm(tiff_paths, desc="  Saving PNGs", leave=False):
        arr    = tifffile.imread(str(tiff_path)).astype(np.float32)
        binary = binarize(arr, threshold)

        rel_path = tiff_path.relative_to(src_root).with_suffix(".png")
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(binary, mode="L").save(out_path)

    if out_path is not None:
        logger.info(f"  Saved {len(tiff_paths)} PNG(s) → {out_path.parent}")


def process_group(
    tiff_paths: list[Path],
    src_root: Path,
    dst_root: Path,
    threshold_scope: str,
) -> None:
    """
    Threshold and save a group of tiff files as PNGs.

    Used only in 'image' scope (per-image threshold).
    For 'group' scope use process_group_with_threshold() instead.
    """
    if not tiff_paths:
        return

    out_path = None
    for tiff_path in tqdm(tiff_paths, desc="  Saving PNGs", leave=False):
        arr       = tifffile.imread(str(tiff_path)).astype(np.float32)
        threshold = compute_image_threshold(arr)
        binary    = binarize(arr, threshold)

        rel_path = tiff_path.relative_to(src_root).with_suffix(".png")
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(binary, mode="L").save(out_path)

    if out_path is not None:
        logger.info(f"  Saved {len(tiff_paths)} PNG(s) → {out_path.parent}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_thresholded_pngs(
    submission_path: str,
    threshold_scope: str = "group",
) -> None:
    """
    Walk anomaly_images/<category>/<split>/ and produce thresholded PNGs
    under anomaly_images_thresholded/<category>/<split>/.

    Threshold is computed per category (all splits combined) in 'group' mode,
    or independently per image in 'image' mode.

    Args:
        submission_path:  Path to the submission directory.
        threshold_scope:  'group' or 'image'.
    """
    submission_path = Path(submission_path)
    src_root = submission_path / "anomaly_images"
    dst_root = submission_path / "anomaly_images_thresholded"

    if not src_root.exists():
        raise FileNotFoundError(f"anomaly_images/ not found in {submission_path}")

    dst_root.mkdir(parents=True, exist_ok=True)

    categories = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not categories:
        logger.warning("No category directories found under anomaly_images/")
        return

    total_files = 0

    for cat_dir in categories:
        splits = sorted([p for p in cat_dir.iterdir() if p.is_dir()])

        if not splits:
            # Flat structure: anomaly_images/<category>/*.tiff
            tiff_paths = sorted(cat_dir.glob("*.tiff"))
            if tiff_paths:
                logger.info(
                    f"Processing category '{cat_dir.name}' "
                    f"({len(tiff_paths)} files, scope={threshold_scope})"
                )
                if threshold_scope == "group":
                    threshold = compute_group_threshold(tiff_paths)
                    process_group_with_threshold(tiff_paths, src_root, dst_root, threshold)
                else:
                    process_group(tiff_paths, src_root, dst_root, threshold_scope)
                total_files += len(tiff_paths)
            continue

        all_tiff_paths: list[Path] = []
        for split_dir in splits:
            tiff_paths = sorted(split_dir.glob("*.tiff"))
            if not tiff_paths:
                logger.warning(
                    f"No .tiff files found in {split_dir.as_posix()}, skipping."
                )
                continue
            all_tiff_paths.extend(tiff_paths)

        if not all_tiff_paths:
            continue

        logger.info(
            f"Processing category '{cat_dir.name}' "
            f"({len(all_tiff_paths)} files across {len(splits)} split(s), "
            f"scope={threshold_scope})"
        )

        if threshold_scope == "group":
            category_threshold = compute_group_threshold(all_tiff_paths)
            process_group_with_threshold(
                all_tiff_paths, src_root, dst_root, category_threshold
            )
        else:
            process_group(all_tiff_paths, src_root, dst_root, threshold_scope)

        total_files += len(all_tiff_paths)

    logger.info(
        f"\nDone. {total_files} PNG(s) saved to: {dst_root.as_posix()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate thresholded PNG anomaly maps from TIFF files."
    )
    parser.add_argument(
        "submission_path",
        type=str,
        help="Path to your submission directory (containing anomaly_images/).",
    )
    parser.add_argument(
        "--threshold-scope",
        type=str,
        choices=["group", "image"],
        default="group",
        help=(
            "How to compute the threshold:\n"
            "  group  — shared threshold per category (all splits) [default]\n"
            "  image  — independent threshold per image"
        ),
    )
    args = parser.parse_args()

    generate_thresholded_pngs(args.submission_path, args.threshold_scope)



