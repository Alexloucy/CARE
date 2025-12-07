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

    # single domain assumed, restriction imposed in front-end
    def forward(self, image, time):
        adapter_ratio = 0.4  
        # Get cls token from DINO
        x_tokens_list = self.dino_model.get_intermediate_layers(image, n=1, return_class_token=True)
        image_features = create_linear_input(x_tokens_list, 1, False)
        # Normalize only at the end; keep raw features here
        base_features = image_features
        if isinstance(time, int):
            time = torch.tensor([time]).to(base_features.device)
        else:
            time = torch.tensor(time).to(base_features.device)
        
        unique_times = torch.unique(time)
        # Mix in-place; avoid cloning to save memory
        mixed_features = base_features

        
        if self.day_night_adapter:
            if time is None:
                raise ValueError("Time information (day/night) must be provided when using day/night adapters.")
            day_adapter = self.adapter_dict["adapter_0_day"]
            night_adapter = self.adapter_dict["adapter_0_night"]
            # iterate through day and night adapters
            for t in unique_times.tolist():
                idx = (time == t).nonzero(as_tuple=False).squeeze(1)
                sub_base_features = base_features.index_select(0, idx)
                if t == 1:  # day
                    sub_adapter_features = day_adapter(sub_base_features)
                else:  # night
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

        # Normalize mixed features for retrieval/metric learning
        mixed_features_norm = torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)
        return mixed_features_norm
    
    def forward_from_raw(self, base_features, time):
        """
        Forward pass that skips the DINO backbone - uses pre-computed dinov3_raw features.
        This is much faster when embeddings are cached from classification.
        
        Args:
            base_features: Pre-computed dinov3_raw features from cache [B, D]
            time: Day/night flags for each sample
        
        Returns:
            Normalized mixed features suitable for ReID
        """
        adapter_ratio = 0.4
        
        if isinstance(time, int):
            time = torch.tensor([time]).to(base_features.device)
        else:
            time = torch.tensor(time).to(base_features.device)
        
        unique_times = torch.unique(time)
        mixed_features = base_features.clone()
        
        if self.day_night_adapter:
            if time is None:
                raise ValueError("Time information (day/night) must be provided when using day/night adapters.")
            day_adapter = self.adapter_dict["adapter_0_day"]
            night_adapter = self.adapter_dict["adapter_0_night"]
            for t in unique_times.tolist():
                idx = (time == t).nonzero(as_tuple=False).squeeze(1)
                sub_base_features = base_features.index_select(0, idx)
                if t == 1:  # day
                    sub_adapter_features = day_adapter(sub_base_features)
                else:  # night
                    sub_adapter_features = night_adapter(sub_base_features)
                sub_mixed_features = (
                    adapter_ratio * sub_adapter_features + (1 - adapter_ratio) * sub_base_features
                )
                mixed_features[idx] = sub_mixed_features
        else:
            adapter = self.adapter_dict["adapter_0"]
            sub_adapter_features = adapter(base_features)
            mixed_features = (
                adapter_ratio * sub_adapter_features + (1 - adapter_ratio) * base_features
            )
        
        # Normalize mixed features for retrieval/metric learning
        mixed_features_norm = torch.nn.functional.normalize(mixed_features, dim=-1, eps=1e-6)
        return mixed_features_norm


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    """Create linear input from DINO intermediate layers"""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
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

    # 255 is for extreme cases, night time photos usually have diff of 0
    if mean_diff < 3 or mean_diff == 255:
        return 0  # night
    else:
        return 1


def load_and_crop_image(image_path, bbox):
    """
    Load image, crop by bbox, and preprocess.
    
    Args:
        image_path: Path to the original image.
        bbox: [x1, y1, x2, y2] coordinates.
    
    Returns:
        (tensor, is_day): Preprocessed image tensor and day/night flag.
    """
    img = Image.open(image_path).convert("RGB")
    
    # Crop the image using bbox
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = img.crop((x1, y1, x2, y2))
    
    # Check day/night on cropped image
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
    fp16_supported = device.type == "cuda" or (device.type == "mps" and torch.backends.mps.is_built())
    return fp16_supported


def get_dino_with_adapter_embedding(model, image, device, is_day_list=None):
    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16, enabled=_fp16_supported(device)):
        image = image.to(device)
        feature = model(image, is_day_list)
        feature = feature.to(device)
    return feature


def compute_embeddings_batched(model, images, times, device, batch_size: int) -> np.ndarray:
    """
    Compute L2-normalized embeddings for all images in mini-batches.

    Args:
        model: DINO+adapter model.
        images: list of tensors, each of shape [1, C, H, W].
        times: list of day/night flags aligned with images.
        device: torch.device to run on.
        batch_size: batch size for inference.

    Returns:
        embeddings: NumPy array of shape [N, D].
    """
    embeddings = []
    total = len(images)
    processed = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_images = images[start:end]
        batch_times = times[start:end]

        # Concatenate into a single batch tensor: [B, C, H, W]
        batch_tensor = torch.cat(batch_images, dim=0)

        batch_embedding = get_dino_with_adapter_embedding(
            model,
            batch_tensor,
            device,
            batch_times,
        )
        # batch_embedding: [B, D]
        batch_np = batch_embedding.cpu().float().numpy()
        embeddings.append(batch_np)

        processed += (end - start)
        print(f"PROCESS: {processed}/{total}", flush=True)

    return np.concatenate(embeddings, axis=0)  # [N, D], L2-normalized


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Build a cosine distance matrix and apply the same per-row masking
    behavior as the original compute_distances(is_duplicate=True).

    Args:
        embeddings: NumPy array of shape [N, D].

    Returns:
        distance_mat: NumPy array of shape [N, N].
    """
    # Cosine similarity matrix: sim[i, j] = <emb_i, emb_j>
    sim_matrix = embeddings @ embeddings.T
    distance_mat = 1.0 - sim_matrix

    # For each row, set the minimum-distance entry to infinity.
    for r in range(distance_mat.shape[0]):
        row = distance_mat[r]
        min_idx = np.argmin(row)
        distance_mat[r, min_idx] = np.inf

    return distance_mat


def process_dist_mat_v2(dist_mat):
    """
    Process the distance matrix to count the number of individuals.
    """
    number_of_images = len(dist_mat)
    keys = np.array([-1] * number_of_images)

    for r in range(len(dist_mat)):
        row = dist_mat[r]
        print(f"Row {r} distances: {row}")
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


def format_output_with_detection_ids(detection_ids, cluster_dict):
    """
    Format output with detection IDs instead of file paths.
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
    db_path = input_data.get('db_path')  # Optional: for embedding cache
    species = input_data.get('species', 'unknown')
    
    # Initialize embedding cache if db_path provided
    cache = None
    if db_path:
        try:
            from db_utils import EmbeddingCache
            cache = EmbeddingCache(db_path)
            print(f"Embedding cache initialized: {db_path}", flush=True)
        except Exception as e:
            print(f"Warning: Could not initialize embedding cache: {e}", flush=True)
            cache = None
    
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
    
    # Set the device to GPU if available, otherwise use CPU.
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using Apple Silicon GPU", flush=True)
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU. Note: Using CPU may be slow.", flush=True)
    
    # Read and import the cfg file.
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file_path)
    cfg.merge_from_list([])
    cfg.freeze()
    
    # Load the model
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
    missing, unexpected = dino_with_adapter.adapter_dict.load_state_dict(
        checkpoint, strict=False
    )
    print("Missing", missing)
    print("Unexpected", unexpected)
    dino_with_adapter = dino_with_adapter.to(DEVICE)
    dino_with_adapter.eval()
    
    print("STATUS: PROCESSING", flush=True)
    
    # Embedding types - using config for consistency
    from config.config import REID_EMBEDDING_PREFIX, RAW_FOR_ADAPTER_TYPE
    reid_embedding_type = f'{REID_EMBEDDING_PREFIX}{species}'
    raw_embedding_type = RAW_FOR_ADAPTER_TYPE  # Set to non-existent name to disable adapter-only path
    
    # Categorize detections into three groups:
    # 1. Already have dinov3_reid (fully cached - just use it)
    # 2. Have dinov3_raw but not dinov3_reid (run adapter only)
    # 3. Have neither (run full model)
    
    cached_reid = {}      # key -> numpy array (final embeddings)
    has_raw = []          # [(idx, det, raw_embedding)] - needs adapter only
    needs_full = []       # [(idx, det)] - needs full model
    detection_ids = []
    
    total = len(detections)
    print(f"Checking cache for {total} detections...", flush=True)
    
    for i, det in enumerate(detections):
        detection_ids.append(det['detection_id'])
        
        if cache and 'image_id' in det:
            image_id = det['image_id']
            bbox = det['bbox']
            key = f"{image_id}:{cache.bbox_to_hash(bbox)}"
            
            # First check: do we have final reid embedding?
            reid_emb = cache.get_embedding(image_id, bbox, reid_embedding_type)
            if reid_emb is not None:
                cached_reid[key] = reid_emb
                continue
            
            # Second check: do we have raw embedding from classification?
            raw_emb = cache.get_embedding(image_id, bbox, raw_embedding_type)
            if raw_emb is not None:
                has_raw.append((i, det, raw_emb))
                continue
        
        # Need full model
        needs_full.append((i, det))
    
    print(f"Cache status: {len(cached_reid)} reid cached, {len(has_raw)} have raw (adapter only), {len(needs_full)} need full model", flush=True)
    
    # Process items that have dinov3_raw (adapter only - FAST)
    raw_embeddings = {}  # idx -> numpy array
    if has_raw:
        print(f"Running adapter on {len(has_raw)} cached raw embeddings...", flush=True)
        
        for batch_start in range(0, len(has_raw), batch_size):
            batch_items = has_raw[batch_start:batch_start + batch_size]
            
            # Prepare batch
            raw_tensors = []
            is_day_list = []
            batch_info = []  # (idx, det)
            
            for idx, det, raw_emb in batch_items:
                raw_tensors.append(torch.from_numpy(raw_emb))
                # Get day/night from image
                try:
                    _, is_day = load_and_crop_image(det['image_path'], det['bbox'])
                    is_day_list.append(is_day)
                except:
                    is_day_list.append(1)  # Default to day if error
                batch_info.append((idx, det))
            
            # Stack and process through adapter only
            batch_tensor = torch.stack(raw_tensors).to(DEVICE)
            
            with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=_fp16_supported(DEVICE)):
                reid_features = dino_with_adapter.forward_from_raw(batch_tensor, is_day_list)
                reid_features_np = reid_features.cpu().float().numpy()
            
            # Save on-the-fly and store results
            items_to_store = []
            for k, (idx, det) in enumerate(batch_info):
                raw_embeddings[idx] = reid_features_np[k]
                if cache and 'image_id' in det:
                    items_to_store.append((det['image_id'], det['bbox'], reid_features_np[k]))
            
            if items_to_store and cache:
                cache.store_embeddings_batch(items_to_store, reid_embedding_type)
            
            processed = min(batch_start + batch_size, len(has_raw))
            print(f"ADAPTER: {processed}/{len(has_raw)}", flush=True)
    
    # Process items that need full model (SLOW)
    full_embeddings = {}  # idx -> numpy array
    if needs_full:
        print(f"Running full model on {len(needs_full)} images...", flush=True)
        
        for batch_start in range(0, len(needs_full), batch_size):
            batch_items = needs_full[batch_start:batch_start + batch_size]
            
            # Load and prepare batch
            images = []
            is_day_list = []
            batch_info = []  # (idx, det)
            
            for idx, det in batch_items:
                try:
                    img, is_day = load_and_crop_image(det['image_path'], det['bbox'])
                    images.append(img)
                    is_day_list.append(is_day)
                    batch_info.append((idx, det))
                except Exception as e:
                    print(f"Error loading {det['image_path']}: {e}", flush=True)
                    # Mark as failed
                    detection_ids[detection_ids.index(det['detection_id'])] = None
            
            if not images:
                continue
            
            # Stack and process through full model
            batch_tensor = torch.cat(images, dim=0).to(DEVICE)
            
            with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=_fp16_supported(DEVICE)):
                reid_features = dino_with_adapter(batch_tensor, is_day_list)
                reid_features_np = reid_features.cpu().float().numpy()
            
            # Save on-the-fly and store results
            items_to_store = []
            for k, (idx, det) in enumerate(batch_info):
                full_embeddings[idx] = reid_features_np[k]
                if cache and 'image_id' in det:
                    items_to_store.append((det['image_id'], det['bbox'], reid_features_np[k]))
            
            if items_to_store and cache:
                cache.store_embeddings_batch(items_to_store, reid_embedding_type)
            
            processed = min(batch_start + batch_size, len(needs_full))
            print(f"PROCESS: {processed}/{len(needs_full)}", flush=True)
    
    # Remove failed detection_ids
    detection_ids = [d for d in detection_ids if d is not None]
    
    # Combine cached and new embeddings in original order
    all_embeddings = []
    for i, det in enumerate(detections):
        if det['detection_id'] not in detection_ids:
            continue  # This detection failed to load
        
        # Check cached_reid first (from previous ReID runs)
        if cache and 'image_id' in det:
            key = f"{det['image_id']}:{cache.bbox_to_hash(det['bbox'])}"
            if key in cached_reid:
                all_embeddings.append(cached_reid[key])
                continue
        
        # Check raw_embeddings (from adapter-only processing)
        if i in raw_embeddings:
            all_embeddings.append(raw_embeddings[i])
            continue
        
        # Check full_embeddings (from full model processing)
        if i in full_embeddings:
            all_embeddings.append(full_embeddings[i])
            continue
    
    if len(all_embeddings) == 0:
        print("No valid embeddings after processing. Exiting.", flush=True)
        output = {"individuals": []}
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("STATUS: DONE", flush=True)
        return
    
    embeddings = np.stack(all_embeddings, axis=0)
    
    distance_mat = compute_distance_matrix(embeddings)
    
    id_dict = process_dist_mat_v2(distance_mat)
    
    # Format output with detection IDs
    output = format_output_with_detection_ids(detection_ids, id_dict)
    
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
