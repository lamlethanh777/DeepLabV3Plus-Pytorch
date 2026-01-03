"""Flask web app for interactive Cityscapes segmentation visualization.

Run with: python app.py
Then open: http://localhost:5000
"""

import io
import os
import sys
import glob
import time
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL.ExifTags import TAGS
from flask import Flask, render_template, request, jsonify, send_file
from torchvision import transforms
from torchvision.datasets import Cityscapes
import base64

# Add parent directory to path to import network module
sys.path.insert(0, str(Path(__file__).parent.parent))
from network import modeling

app = Flask(__name__)

# Configuration
DATASET_CONFIG = {
    "cityscapes": {"num_classes": 19, "dataset_name": "Cityscapes"},
    "voc": {"num_classes": 21, "dataset_name": "Pascal VOC"}
}
CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default to Cityscapes for backward compatibility
NUM_CLASSES = 19

# Global model cache
loaded_models = {}

# Prediction cache for real-time overlay adjustment
prediction_cache = {
    "image": None,
    "pred": None,
    "image_b64": None,
}


def _build_label_mappings():
    """Build label mappings from Cityscapes.classes."""
    id_to_train_id = np.full(256, 255, dtype=np.int64)
    train_id_to_label_id = np.zeros(256, dtype=np.uint8)
    class_names = [None] * NUM_CLASSES
    class_colors = {}

    for c in Cityscapes.classes:
        if 0 <= c.id < 256:
            train_id = c.train_id if c.train_id != -1 else 255
            id_to_train_id[c.id] = train_id
            if 0 <= train_id < NUM_CLASSES:
                train_id_to_label_id[train_id] = c.id
                class_names[train_id] = c.name
                class_colors[train_id] = c.color

    return id_to_train_id, train_id_to_label_id, class_names, class_colors


def _voc_cmap(N=256):
    """Generate VOC colormap."""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


ID_TO_TRAIN_ID, TRAIN_ID_TO_LABEL_ID, CLASS_NAMES, CLASS_COLORS = _build_label_mappings()

# Create color palette for Cityscapes (trainId -> RGB)
COLOR_PALETTE_CITYSCAPES = np.zeros((256, 3), dtype=np.uint8)
for train_id, color in CLASS_COLORS.items():
    COLOR_PALETTE_CITYSCAPES[train_id] = color
COLOR_PALETTE_CITYSCAPES[255] = (0, 0, 0)  # Void class

# Create color palette for VOC
COLOR_PALETTE_VOC = _voc_cmap(256)

# Default to Cityscapes for backward compatibility
COLOR_PALETTE = COLOR_PALETTE_CITYSCAPES


def get_system_info():
    """Get system and environment information."""
    info = {
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "num_classes": NUM_CLASSES,
    }
    if torch.cuda.is_available():
        info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
    return info


def extract_image_metadata(image, file_obj=None):
    """Extract metadata from an image."""
    metadata = {
        "width": image.width,
        "height": image.height,
        "aspect_ratio": f"{image.width / image.height:.2f}",
        "mode": image.mode,
        "format": image.format or "Unknown",
        "megapixels": f"{(image.width * image.height) / 1e6:.2f} MP",
    }
    
    # Try to get file size
    if file_obj:
        file_obj.seek(0, 2)  # Seek to end
        metadata["file_size"] = f"{file_obj.tell() / 1024:.1f} KB"
        file_obj.seek(0)  # Reset to beginning
    
    # Try to extract EXIF data
    try:
        exif = image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ["Make", "Model", "DateTime", "ExposureTime", "FNumber", "ISOSpeedRatings"]:
                    metadata[f"exif_{tag}"] = str(value)
    except:
        pass
    
    return metadata


def get_class_distribution(mask, dataset="cityscapes", num_classes=19):
    """Get distribution of classes in the segmentation mask."""
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    # Get appropriate color palette
    palette = COLOR_PALETTE_VOC if dataset == "voc" else COLOR_PALETTE_CITYSCAPES
    
    distribution = []
    for cls_id, count in zip(unique, counts):
        if cls_id < num_classes:
            percentage = (count / total_pixels) * 100
            # Get color from palette
            color = palette[cls_id].tolist() if cls_id < len(palette) else [0, 0, 0]
            # Get name from Cityscapes classes or use generic name for VOC
            if dataset == "cityscapes" and cls_id < len(CLASS_NAMES):
                name = CLASS_NAMES[cls_id]
            else:
                name = f"Class {cls_id}"
            
            distribution.append({
                "id": int(cls_id),
                "name": name,
                "pixels": int(count),
                "percentage": round(percentage, 2),
                "color": color
            })
    
    # Sort by percentage descending
    distribution.sort(key=lambda x: x["percentage"], reverse=True)
    return distribution


def get_available_checkpoints():
    """Get list of available checkpoint files with metadata.
    
    Returns:
        List of dicts with keys: name, path, dataset, config_path, experiment
    """
    checkpoints = []
    
    # Scan experiment folders (e0, e1, e2, etc.)
    if EXPERIMENT_DIR.exists():
        for exp_dir in sorted(EXPERIMENT_DIR.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith('e'):
                continue
            
            exp_checkpoints_dir = exp_dir / "checkpoints"
            if not exp_checkpoints_dir.exists():
                continue
            
            # Find all .pth files (excluding archive folder)
            for pth_file in exp_checkpoints_dir.rglob("*.pth"):
                # Skip archive folders
                if "archive" in pth_file.parts:
                    continue
                
                # Extract dataset from filename
                filename = pth_file.stem
                dataset = "voc" if "_voc_" in filename else "cityscapes"
                
                # Find matching config file
                config_path = None
                for yaml_file in exp_dir.glob("*.yaml"):
                    yaml_stem = yaml_file.stem.lower()
                    if dataset in yaml_stem:
                        config_path = yaml_file
                        break
                
                # Create display name
                display_name = f"[{exp_dir.name}] {pth_file.name}"
                
                checkpoints.append({
                    "name": display_name,
                    "path": str(pth_file),
                    "dataset": dataset,
                    "config_path": str(config_path) if config_path else None,
                    "experiment": exp_dir.name
                })
    
    # Also check main checkpoints folder
    if CHECKPOINTS_DIR.exists():
        for pth_file in CHECKPOINTS_DIR.glob("*.pth"):
            filename = pth_file.stem
            dataset = "voc" if "_voc_" in filename else "cityscapes"
            
            checkpoints.append({
                "name": pth_file.name,
                "path": str(pth_file),
                "dataset": dataset,
                "config_path": None,
                "experiment": "main"
            })
    
    # Check parent folder for any .pth files
    parent_dir = Path(__file__).parent.parent
    for pth_file in parent_dir.glob("*.pth"):
        filename = pth_file.stem
        dataset = "voc" if "_voc_" in filename else "cityscapes"
        
        checkpoints.append({
            "name": pth_file.name,
            "path": str(pth_file),
            "dataset": dataset,
            "config_path": None,
            "experiment": "root"
        })
    
    return checkpoints


def get_checkpoint_info(checkpoint_name):
    """Get checkpoint info including path and config.
    
    Args:
        checkpoint_name: Display name of checkpoint
        
    Returns:
        Dict with checkpoint info or None if not found
    """
    checkpoints = get_available_checkpoints()
    for cp in checkpoints:
        if cp["name"] == checkpoint_name:
            return cp
    return None


def load_model(checkpoint_name, num_classes=19):
    """Load model from checkpoint (with caching).
    
    Args:
        checkpoint_name: Display name of checkpoint
        num_classes: Number of classes for the model
        
    Returns:
        Loaded model
    """
    cache_key = f"{checkpoint_name}_{num_classes}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]
    
    checkpoint_info = get_checkpoint_info(checkpoint_name)
    if checkpoint_info is None:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")
    
    checkpoint_path = Path(checkpoint_info["path"])
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load config to get model architecture details
    config_data = None
    if checkpoint_info["config_path"]:
        config_data = load_config(checkpoint_info["config_path"])
    
    # Determine model architecture - ALWAYS trust the config file if available
    model_name = None
    output_stride = 16
    
    if config_data and "model" in config_data:
        # Use the model from config - this is the ground truth
        model_name = config_data["model"]
        output_stride = config_data.get("output_stride", 16)
        print(f"Using model from config: {model_name}")
    
    # Only fall back to filename inference if no config available
    if model_name is None:
        model_name = "deeplabv3plus_mobilenet_v3_large"  # default
        filename = checkpoint_path.stem.lower()
        if "mobilenet_v3" in filename or "mobilenet_v3_large" in filename:
            if "attention" in filename:
                model_name = "deeplabv3plus_mobilenet_v3_large_attention"
            else:
                model_name = "deeplabv3plus_mobilenet_v3_large"
        elif "mobilenet" in filename and "attention" in filename:
            model_name = "deeplabv3plus_mobilenet_attention"
        elif "mobilenet" in filename and "epsa" in filename:
            model_name = "deeplabv3plus_mobilenet_epsa"
        elif "mobilenet" in filename:
            model_name = "deeplabv3plus_mobilenet"
        print(f"Inferred model from filename: {model_name}")
    
    # Create model using the repository's model factory (same as main.py)
    try:
        # Use the same approach as main.py - call the function from modeling.__dict__
        if model_name in modeling.__dict__:
            model = modeling.__dict__[model_name](
                num_classes=num_classes,
                output_stride=output_stride
            )
            print(f"Created model: {model_name}")
        else:
            print(f"Warning: Model {model_name} not found in modeling.__dict__, using default")
            model = modeling.deeplabv3plus_mobilenet(
                num_classes=num_classes,
                output_stride=output_stride
            )
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        # Fallback to default model
        model = modeling.deeplabv3plus_mobilenet(
            num_classes=num_classes,
            output_stride=output_stride
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats (same as main.py)
    if isinstance(checkpoint, dict):
        # Main.py uses "model_state" key
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
    else:
        # Checkpoint is directly the state dict
        state_dict = checkpoint
    
    # Load checkpoint weights (same as main.py, but with strict=False for flexibility)
    try:
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint successfully")
    except Exception as e:
        print(f"Error loading with strict=True, trying strict=False: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Cache the model
    loaded_models[cache_key] = model
    
    return model


def load_config(config_path):
    """Load YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict with config data or None if file doesn't exist
    """
    if not config_path or not Path(config_path).exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None


def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def predict(model, image):
    """Run prediction on image. Returns prediction and timing info."""
    # Preprocess
    preprocess_start = time.time()
    input_tensor = preprocess_image(image).to(DEVICE)
    preprocess_time = time.time() - preprocess_start
    
    # Inference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        # Handle both dict output (torchvision) and direct tensor output (custom models)
        if isinstance(output, dict):
            output = output["out"]
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    
    timing = {
        "preprocess_ms": round(preprocess_time * 1000, 2),
        "inference_ms": round(inference_time * 1000, 2),
        "total_ms": round((preprocess_time + inference_time) * 1000, 2),
        "fps": round(1.0 / (preprocess_time + inference_time), 1) if (preprocess_time + inference_time) > 0 else 0
    }
    
    return pred, timing


def colorize_mask(mask, dataset="cityscapes"):
    """Convert mask to RGB color image."""
    palette = COLOR_PALETTE_VOC if dataset == "voc" else COLOR_PALETTE_CITYSCAPES
    colored = palette[mask.astype(np.uint8)]
    return Image.fromarray(colored, mode="RGB")


def create_overlay(image, mask, alpha=0.5, dataset="cityscapes"):
    """Create overlay of mask on original image."""
    image = image.convert("RGB")
    colored_mask = colorize_mask(mask, dataset=dataset)
    colored_mask = colored_mask.resize(image.size, Image.NEAREST)
    
    # Blend
    overlay = Image.blend(image, colored_mask, alpha)
    return overlay


def compute_metrics(pred, gt, dataset="cityscapes", num_classes=19):
    """Compute segmentation metrics between prediction and ground truth."""
    # For Cityscapes: convert label IDs to train IDs
    # For VOC: GT already contains class IDs (0-20), no conversion needed
    if dataset == "cityscapes":
        gt_train = ID_TO_TRAIN_ID[gt.astype(np.int64)]
    else:
        # VOC ground truth is already in class ID format
        gt_train = gt.astype(np.int64)
    
    # Flatten
    pred_flat = pred.flatten()
    gt_flat = gt_train.flatten()
    
    # Mask for valid pixels (ignore void class 255)
    valid_mask = (gt_flat != 255) & (gt_flat < num_classes)
    pred_valid = pred_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]
    
    if len(gt_valid) == 0:
        return None
    
    # Overall accuracy
    accuracy = (pred_valid == gt_valid).sum() / len(gt_valid)
    
    # Per-class metrics
    class_metrics = []
    for cls_id in range(num_classes):
        pred_cls = (pred_valid == cls_id)
        gt_cls = (gt_valid == cls_id)
        
        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()
        
        # IoU
        iou = intersection / (union + 1e-10) if union > 0 else 0
        
        # Dice
        dice = 2 * intersection / (pred_cls.sum() + gt_cls.sum() + 1e-10) if (pred_cls.sum() + gt_cls.sum()) > 0 else 0
        
        # Class accuracy
        cls_acc = intersection / (gt_cls.sum() + 1e-10) if gt_cls.sum() > 0 else None
        
        if gt_cls.sum() > 0:  # Only include classes present in GT
            class_metrics.append({
                "name": CLASS_NAMES[cls_id],
                "iou": float(iou),
                "dice": float(dice),
                "accuracy": float(cls_acc) if cls_acc is not None else None,
                "gt_pixels": int(gt_cls.sum()),
                "pred_pixels": int(pred_cls.sum()),
            })
    
    # Mean IoU (only for classes present in GT)
    if class_metrics:
        mean_iou = np.mean([m["iou"] for m in class_metrics])
        mean_dice = np.mean([m["dice"] for m in class_metrics])
    else:
        mean_iou = 0
        mean_dice = 0
    
    return {
        "overall_accuracy": float(accuracy),
        "mean_iou": float(mean_iou),
        "mean_dice": float(mean_dice),
        "class_metrics": class_metrics,
    }


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def create_grid_image(images, labels, grid_size=(2, 2), cell_size=(512, 256), label_height=30):
    """Create a grid image with labels under each image.
    
    Args:
        images: List of PIL Images (can contain None for empty cells)
        labels: List of labels for each image
        grid_size: (rows, cols) tuple
        cell_size: (width, height) for each cell's image area
        label_height: Height reserved for label text
    
    Returns:
        PIL Image of the grid
    """
    from PIL import ImageDraw, ImageFont
    
    rows, cols = grid_size
    cell_w, cell_h = cell_size
    total_cell_h = cell_h + label_height
    
    # Create grid canvas
    grid_w = cols * cell_w
    grid_h = rows * total_cell_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        
        x_offset = col * cell_w
        y_offset = row * total_cell_h
        
        if img is not None:
            # Resize image to fit cell while maintaining aspect ratio
            img_resized = img.copy()
            img_resized.thumbnail((cell_w - 10, cell_h - 10), Image.LANCZOS)
            
            # Center image in cell
            paste_x = x_offset + (cell_w - img_resized.width) // 2
            paste_y = y_offset + (cell_h - img_resized.height) // 2
            
            grid.paste(img_resized, (paste_x, paste_y))
        
        # Draw label centered below image
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (cell_w - text_w) // 2
        text_y = y_offset + cell_h + 5
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
    
    return grid


@app.route("/")
def index():
    """Main page."""
    checkpoints = get_available_checkpoints()
    return render_template(
        "index.html", 
        checkpoints=checkpoints, 
        datasets=DATASET_CONFIG,
        class_names=CLASS_NAMES, 
        class_colors=CLASS_COLORS
    )


@app.route("/predict", methods=["POST"])
def run_prediction():
    """Run prediction on uploaded image."""
    try:
        total_start = time.time()
        
        # Get checkpoint
        checkpoint_name = request.form.get("checkpoint")
        if not checkpoint_name:
            return jsonify({"error": "No checkpoint selected"}), 400
        
        # Get checkpoint info
        checkpoint_info = get_checkpoint_info(checkpoint_name)
        if not checkpoint_info:
            return jsonify({"error": f"Checkpoint not found: {checkpoint_name}"}), 400
        
        # Get dataset and num_classes
        dataset = checkpoint_info["dataset"]
        num_classes = DATASET_CONFIG[dataset]["num_classes"]
        
        # Load config if available
        config_data = None
        if checkpoint_info["config_path"]:
            config_data = load_config(checkpoint_info["config_path"])
        
        # Get overlay alpha
        overlay_alpha = float(request.form.get("overlay_alpha", 0.5))
        
        # Get image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files["image"]
        
        # Extract metadata before converting
        image_file.seek(0)
        image_raw = Image.open(image_file)
        image_metadata = extract_image_metadata(image_raw, image_file)
        image_metadata["filename"] = image_file.filename
        
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        original_size = image.size
        
        # Load model and predict
        model_load_start = time.time()
        model = load_model(checkpoint_name, num_classes)
        model_load_time = time.time() - model_load_start
        
        pred, timing = predict(model, image)
        
        # Resize prediction to original size
        postprocess_start = time.time()
        pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST))
        
        # Cache prediction for real-time overlay adjustment
        prediction_cache["image"] = image.copy()
        prediction_cache["pred"] = pred_resized.copy()
        prediction_cache["image_b64"] = image_to_base64(image)
        
        # Create visualizations
        mask_colored = colorize_mask(pred_resized, dataset=dataset)
        overlay = create_overlay(image, pred_resized, alpha=overlay_alpha, dataset=dataset)
        postprocess_time = time.time() - postprocess_start
        
        # Get class distribution
        class_distribution = get_class_distribution(pred_resized, dataset=dataset, num_classes=num_classes)
        
        # Build timing info
        timing["model_load_ms"] = round(model_load_time * 1000, 2) if model_load_time > 0.01 else "cached"
        timing["postprocess_ms"] = round(postprocess_time * 1000, 2)
        timing["total_request_ms"] = round((time.time() - total_start) * 1000, 2)
        
        result = {
            "mask": image_to_base64(mask_colored),
            "overlay": image_to_base64(overlay),
            "pred_shape": list(pred_resized.shape),
            "image_metadata": image_metadata,
            "timing": timing,
            "class_distribution": class_distribution,
            "system_info": get_system_info(),
            "model_name": checkpoint_name,
            "dataset": dataset,
            "num_classes": num_classes,
            "config": config_data,
            "experiment": checkpoint_info["experiment"],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Process ground truth if provided
        if "groundtruth" in request.files and request.files["groundtruth"].filename:
            gt_file = request.files["groundtruth"]
            gt = np.array(Image.open(gt_file))
            
            # Resize GT to match prediction if needed
            if gt.shape != pred_resized.shape:
                gt = np.array(Image.fromarray(gt).resize(original_size, Image.NEAREST))
            
            # Compute metrics
            metrics = compute_metrics(pred_resized, gt, dataset=dataset, num_classes=num_classes)
            if metrics:
                result["metrics"] = metrics
            
            # Create GT visualization
            # For Cityscapes: convert label IDs to train IDs
            # For VOC: GT already contains class IDs
            if dataset == "cityscapes":
                gt_train = ID_TO_TRAIN_ID[gt.astype(np.int64)].astype(np.uint8)
            else:
                gt_train = gt.astype(np.uint8)
            
            gt_colored = colorize_mask(gt_train, dataset=dataset)
            result["groundtruth_colored"] = image_to_base64(gt_colored)
            
            # Add GT class distribution
            result["gt_class_distribution"] = get_class_distribution(gt_train, dataset=dataset, num_classes=num_classes)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/class_legend")
def class_legend():
    """Return class legend data."""
    legend = []
    for i, name in enumerate(CLASS_NAMES):
        if name:
            color = CLASS_COLORS.get(i, (0, 0, 0))
            legend.append({
                "id": i,
                "name": name,
                "color": f"rgb({color[0]}, {color[1]}, {color[2]})"
            })
    return jsonify(legend)


@app.route("/checkpoint_config", methods=["POST"])
def checkpoint_config():
    """Get config data for a checkpoint."""
    try:
        checkpoint_name = request.json.get("checkpoint")
        if not checkpoint_name:
            return jsonify({"error": "No checkpoint specified"}), 400
        
        checkpoint_info = get_checkpoint_info(checkpoint_name)
        if not checkpoint_info:
            return jsonify({"error": "Checkpoint not found"}), 404
        
        config_data = None
        if checkpoint_info["config_path"]:
            config_data = load_config(checkpoint_info["config_path"])
        
        return jsonify({
            "checkpoint": checkpoint_info,
            "config": config_data,
            "dataset": checkpoint_info["dataset"],
            "num_classes": DATASET_CONFIG[checkpoint_info["dataset"]]["num_classes"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/system_info")
def system_info():
    """Return system information."""
    return jsonify(get_system_info())


@app.route("/update_overlay", methods=["POST"])
def update_overlay():
    """Update overlay with new alpha using cached prediction."""
    try:
        if prediction_cache["image"] is None or prediction_cache["pred"] is None:
            return jsonify({"error": "No cached prediction available"}), 400
        
        alpha = float(request.json.get("alpha", 0.5))
        dataset = request.json.get("dataset", "cityscapes")
        
        # Create new overlay with updated alpha
        overlay = create_overlay(prediction_cache["image"], prediction_cache["pred"], alpha=alpha, dataset=dataset)
        
        return jsonify({
            "overlay": image_to_base64(overlay),
            "alpha": alpha
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate_diff", methods=["POST"])
def generate_diff():
    """Generate difference image between prediction and ground truth."""
    try:
        # Get checkpoint
        checkpoint_name = request.form.get("checkpoint")
        if not checkpoint_name:
            return jsonify({"error": "No checkpoint selected"}), 400
        
        # Get checkpoint info
        checkpoint_info = get_checkpoint_info(checkpoint_name)
        if not checkpoint_info:
            return jsonify({"error": "Checkpoint not found"}), 404
        
        dataset = checkpoint_info["dataset"]
        num_classes = DATASET_CONFIG[dataset]["num_classes"]
        
        # Get image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        if "groundtruth" not in request.files:
            return jsonify({"error": "No ground truth uploaded"}), 400
        
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        original_size = image.size
        
        # Load model and predict
        model = load_model(checkpoint_name, num_classes)
        pred, _ = predict(model, image)
        
        # Resize prediction to original size
        pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST))
        
        # Load ground truth
        gt_file = request.files["groundtruth"]
        gt = np.array(Image.open(gt_file))
        
        if gt.shape != pred_resized.shape:
            gt = np.array(Image.fromarray(gt).resize(original_size, Image.NEAREST))
        
        # Convert GT appropriately based on dataset
        if dataset == "cityscapes":
            gt_train = ID_TO_TRAIN_ID[gt.astype(np.int64)]
        else:
            gt_train = gt.astype(np.int64)
        
        # Create diff image
        # Green = correct, Red = incorrect, Gray = ignored (void class)
        diff_img = np.zeros((*pred_resized.shape, 3), dtype=np.uint8)
        
        # Correct predictions (green)
        correct_mask = (pred_resized == gt_train) & (gt_train != 255)
        diff_img[correct_mask] = [81, 207, 102]  # Green
        
        # Incorrect predictions (red)
        incorrect_mask = (pred_resized != gt_train) & (gt_train != 255)
        diff_img[incorrect_mask] = [255, 107, 107]  # Red
        
        # Ignored pixels (gray)
        ignored_mask = (gt_train == 255)
        diff_img[ignored_mask] = [73, 80, 87]  # Gray
        
        diff_pil = Image.fromarray(diff_img, mode="RGB")
        
        return jsonify({
            "diff": image_to_base64(diff_pil)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/export", methods=["POST"])
def export_grid():
    """Export results as a 2x2 grid image."""
    try:
        # Get checkpoint
        checkpoint_name = request.form.get("checkpoint")
        if not checkpoint_name:
            return jsonify({"error": "No checkpoint selected"}), 400
        
        # Get checkpoint info
        checkpoint_info = get_checkpoint_info(checkpoint_name)
        if not checkpoint_info:
            return jsonify({"error": "Checkpoint not found"}), 404
        
        dataset = checkpoint_info["dataset"]
        num_classes = DATASET_CONFIG[dataset]["num_classes"]
        
        # Get image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        original_size = image.size
        
        # Load model and predict
        model = load_model(checkpoint_name, num_classes)
        pred, _ = predict(model, image)
        
        # Resize prediction to original size
        pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(original_size, Image.NEAREST))
        
        # Create visualizations
        mask_colored = colorize_mask(pred_resized, dataset=dataset)
        overlay = create_overlay(image, pred_resized, alpha=0.5)
        
        # Prepare images and labels for grid
        images = [image, mask_colored, overlay, None]
        labels = ["Original Image", "Predicted Mask", "Overlay", "Ground Truth (N/A)"]
        
        # Process ground truth if provided
        if "groundtruth" in request.files and request.files["groundtruth"].filename:
            gt_file = request.files["groundtruth"]
            gt = np.array(Image.open(gt_file))
            
            # Resize GT to match prediction if needed
            if gt.shape != pred_resized.shape:
                gt = np.array(Image.fromarray(gt).resize(original_size, Image.NEAREST))
            
            # Create GT visualization
            if dataset == "cityscapes":
                gt_train = ID_TO_TRAIN_ID[gt.astype(np.int64)].astype(np.uint8)
            else:
                gt_train = gt.astype(np.uint8)
            gt_colored = colorize_mask(gt_train, dataset=dataset)
            images[3] = gt_colored
            labels[3] = "Ground Truth"
        
        # Create grid
        grid = create_grid_image(images, labels)
        
        # Return as downloadable image
        buffer = io.BytesIO()
        grid.save(buffer, format="PNG")
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype="image/png",
            as_attachment=True,
            download_name="segmentation_result.png"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Available checkpoints: {get_available_checkpoints()}")
    print("\nStarting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
