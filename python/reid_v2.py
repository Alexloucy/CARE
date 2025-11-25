"""
ReID v2 - JSON-based input/output for direct database integration.

Input JSON format:
{
    "detections": [
        {
            "detection_id": 42,
            "image_path": "/path/to/image.jpg",
            "bbox": [x1, y1, x2, y2]
        },
        ...
    ],
    "output_path": "/path/to/output.json"
}

Output JSON format:
{
    "individuals": [
        {
            "name": "ID-0",
            "detection_ids": [42, 43, 88]
        },
        ...
    ]
}

Usage:
    python main.py reid_v2 /path/to/input.json
"""

import json
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torch.amp import autocast

from config import cfg
from datetime import datetime
from PIL import Image
from pathlib import Path


class Adapter(nn.Module):
    def __init__(self, channel_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.ReLU(),  
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomDino(nn.Module):
    def __init__(self, cfg, dino_model, domains):
        super().__init__()
        self.cfg = cfg
        self.dino_model = dino_model

        output_dim = self.dino_model.embed_dim
        self.adapter_dict = nn.ModuleDict()
        self.classifier_dict = nn.ModuleDict()

        self.day_night_adapter = cfg.MODEL.Day_Night_Adapter

        for i, domain in enumerate(domains):
            if self.day_night_adapter:
                self.adapter_dict[f"adapter_{i}_day"] = Adapter(output_dim, 4)
                self.adapter_dict[f"adapter_{i}_night"] = Adapter(output_dim, 4)
            else:
                self.adapter_dict[f"adapter_{i}"] = Adapter(output_dim, 4)
        self.domains = domains

    def forward(self, image, time):
        adapter_ratio = 0.4  
        x_tokens_list = self.dino_model.get_intermediate_layers(image, n=1, return_class_token=True)
        image_features = create_linear_input(x_tokens_list, 1, False)
        base_features = image_features
        if isinstance(time, int):
            time = torch.tensor([time]).to(base_features.device)
        else:
            time = torch.tensor(time).to(base_features.device)
        
        unique_times = torch.unique(time)
        mixed_features = base_features

        if self.day_night_adapter:
            if time is None:
                raise ValueError("Time information (day/night) must be provided when using day/night adapters.")
            day_adapter = self.adapter_dict["adapter_0_day"]
            night_adapter = self.adapter_dict["adapter_0_night"]
            for t in unique_times.tolist():
                idx = (time == t).nonzero(as_tuple=False).squeeze(1)
                sub_base_features = base_features.index_select(0, idx)
                if t == 1:
                    sub_adapter_features = day_adapter(sub_base_features)
                else:
                    sub_adapter_features = night_adapter(sub_base_features)
                sub_mixed_features = (
                    adapter_ratio * sub_adapter_features + (1 - adapter_ratio) * sub_base_features
                )
                mixed_features[idx] = sub_mixed_features
        else:
            adapter = self.adapter_dict["adapter_0"]
            sub_adapter_features = adapter(base_features)
            sub_mixed_features = (
                adapter_ratio * sub_adapter_features + (1 - adapter_ratio) * base_features
            )

        mixed_features_norm = torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)
        return mixed_features_norm


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output


def check_day_night(img):
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    dff_rg = np.median(np.abs(r - g))
    dff_rb = np.median(np.abs(r - b))
    dff_gb = np.median(np.abs(g - b))
    mean_diff = np.max([dff_rg, dff_rb, dff_gb])
    if mean_diff < 3 or mean_diff == 255:
        return 0  # night
    else:
        return 1  # day


def load_and_crop_image(image_path: str, bbox: list):
    """
    Load image, crop to bbox, and preprocess for model.
    bbox format: [x1, y1, x2, y2]
    """
    img = Image.open(image_path).convert("RGB")
    
    # Crop to bbox
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img.crop((x1, y1, x2, y2))
    
    is_day = check_day_night(cropped_img)

    image_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    image = image_transforms(cropped_img)
    image = image.unsqueeze(0)
    return image, is_day


def _fp16_supported(device):
    return device.type == "cuda" or (device.type == "mps" and torch.backends.mps.is_built())


def get_embedding(model, image, device, is_day):
    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16, enabled=_fp16_supported(device)):
        image = image.to(device)
        feature = model(image, is_day)
        feature = feature.to(device)
    return feature


def compute_embeddings_batched(model, images, times, device, batch_size: int) -> np.ndarray:
    """
    Compute L2-normalized embeddings for all images in mini-batches.
    """
    embeddings = []
    total = len(images)
    processed = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = images[start:end]
        batch_times = times[start:end]

        batch_tensor = torch.cat(batch_images, dim=0)

        batch_embedding = get_embedding(
            model,
            batch_tensor,
            device,
            batch_times,
        )
        batch_np = batch_embedding.cpu().float().numpy()
        embeddings.append(batch_np)

        processed += (end - start)
        print(f"PROCESS: {processed}/{total}", flush=True)

    return np.concatenate(embeddings, axis=0)


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Build a cosine distance matrix.
    """
    sim_matrix = embeddings @ embeddings.T
    distance_mat = 1.0 - sim_matrix

    # For each row, set the minimum-distance entry (self) to infinity
    for r in range(distance_mat.shape[0]):
        row = distance_mat[r]
        min_idx = np.argmin(row)
        distance_mat[r, min_idx] = np.inf

    return distance_mat


def process_dist_mat_v2(dist_mat):
    """
    Process the distance matrix to cluster individuals.
    """
    number_of_images = len(dist_mat)
    keys = np.array([-1] * number_of_images)

    for r in range(len(dist_mat)):
        row = dist_mat[r]
        min_dist = np.min(row)
        candidates_bool = np.abs(row - min_dist) <= 0.00065
        candidates_index = np.where(candidates_bool)[0]
        candidates_key = keys[candidates_index]
        current_counter = np.max(keys)

        if keys[r] != -1:
            keys[candidates_index] = keys[r]
        elif keys[r] == -1 and np.all(candidates_key == -1):
            keys[r] = current_counter + 1
            keys[candidates_index] = current_counter + 1
        elif keys[r] == -1 and np.any(candidates_key != -1):
            min_pos_key = np.min(candidates_key[candidates_key != -1])
            selected_indices = candidates_index[np.where(candidates_key != min_pos_key)[0]]
            keys[r] = min_pos_key
            keys[selected_indices] = min_pos_key

    aid = 0
    output_dict = dict()
    min_key, max_key = np.min(keys), np.max(keys)
    for k in range(min_key, max_key + 1):
        if k in keys:
            if aid not in output_dict:
                output_dict[aid] = list(np.where(keys == k)[0])
                aid += 1
    return output_dict


def format_output_with_detection_ids(detection_ids: list, cluster_dict: dict) -> dict:
    """
    Convert cluster indices to detection IDs.
    """
    individuals = []
    for cluster_id, indices in cluster_dict.items():
        individuals.append({
            "name": f"ID-{cluster_id}",
            "detection_ids": [detection_ids[idx] for idx in indices]
        })
    return {"individuals": individuals}


def run(input_json_path: str, batch_size: int = 4):
    """
    Main entry point for reid_v2.
    """
    print("STATUS: BEGIN", flush=True)
    
    # Load input JSON
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    detections = input_data['detections']
    output_path = input_data['output_path']
    
    if len(detections) == 0:
        print("No detections provided. Exiting.", flush=True)
        output = {"individuals": []}
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("STATUS: DONE", flush=True)
        return
    
    if len(detections) == 1:
        # Single detection = single individual
        output = {"individuals": [{"name": "ID-0", "detection_ids": [detections[0]['detection_id']]}]}
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("STATUS: DONE", flush=True)
        return
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dino_backbone_path = os.path.join(script_dir, "models", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
    adapter_path = os.path.join(script_dir, "models", "DinoAdapter_Stoat_day_night_mixed_precision.pth.tar25")
    cfg_file_path = os.path.join(script_dir, "models", "dinoadapter_inference.yaml")
    
    # Device selection
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using Apple Silicon GPU", flush=True)
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU", flush=True)
    
    # Load config
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file_path)
    cfg.merge_from_list([])
    cfg.freeze()
    
    # Load model
    print("Loading model...", flush=True)
    repo = os.path.join(script_dir, "dinov3")
    dino_model = torch.hub.load(
        repo, 
        'dinov3_vith16plus', 
        source='local', 
        weights=dino_backbone_path
    )
    dino_model = dino_model.to(DEVICE)
    dino_model.eval()

    dino_with_adapter = CustomDino(
        cfg,
        dino_model=dino_model,
        domains=[0],
    )
    checkpoint = torch.load(adapter_path, map_location="cpu")
    for k, v in list(checkpoint.items()):
        if k.startswith("adapter_dict."):
            checkpoint[k[len("adapter_dict."):]] = v
            del checkpoint[k]
    dino_with_adapter.adapter_dict.load_state_dict(checkpoint, strict=False)
    dino_with_adapter = dino_with_adapter.to(DEVICE)
    dino_with_adapter.eval()
    
    print("STATUS: PROCESSING", flush=True)
    
    # Load and crop images directly from paths (no file copying!)
    detection_ids = []
    images = []
    is_day_list = []
    
    total = len(detections)
    print(f"Loading {total} images...", flush=True)
    
    for i, det in enumerate(detections):
        detection_ids.append(det['detection_id'])
        try:
            img, is_day = load_and_crop_image(det['image_path'], det['bbox'])
            images.append(img)
            is_day_list.append(is_day)
        except Exception as e:
            print(f"Error loading {det['image_path']}: {e}", flush=True)
            # Skip this detection
            detection_ids.pop()
    
    if len(images) == 0:
        print("No valid images after loading. Exiting.", flush=True)
        output = {"individuals": []}
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("STATUS: DONE", flush=True)
        return
    
    # Compute embeddings
    embeddings = compute_embeddings_batched(
        dino_with_adapter,
        images,
        is_day_list,
        DEVICE,
        batch_size,
    )
    
    # Compute distance matrix and cluster
    distance_mat = compute_distance_matrix(embeddings)
    cluster_dict = process_dist_mat_v2(distance_mat)
    
    # Format output with detection IDs
    output = format_output_with_detection_ids(detection_ids, cluster_dict)
    
    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Identified {len(output['individuals'])} individuals", flush=True)
    print("STATUS: DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reid_v2.py <input_json_path> [batch_size]")
        sys.exit(1)
    
    input_json_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(input_json_path, batch_size)
