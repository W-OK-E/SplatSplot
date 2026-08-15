import os
import glob
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import sys

# SAM2 imports
# We need to add sam2_repo/sam2-main to sys.path so we can import sam2 if it's not installed globally
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam2_repo', 'sam2-main')))
from sam2.build_sam import build_sam2_video_predictor

DATA_ROOT = os.environ.get("DATA_ROOT", ".")

# Our carefully curated prompt dictionary
PROMPTS = {
    "aloe": "aloe plant.",
    "art": "concrete sculpture.",
    "car": "parked car.",
    "century": "century plant.",
    "flowers": "flowers in vase.",
    "garbage": "blue dumpster.",
    "picnic": "wooden picnic table.",
    "pikachu": "pikachu plush toy.",
    "pipe": "red water valve.",
    "plant": "potted plant.",
    "roses": "red roses.",
    "table": "white round table."
}

def get_largest_box(results):
    if len(results["boxes"]) == 0:
        return None
    boxes = results["boxes"].cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_idx = np.argmax(areas)
    return boxes[largest_idx]

def main():
    test_scene = sys.argv[1] if len(sys.argv) > 1 else None

    # Determine device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        # SAM2 sometimes struggles with MPS bfloat16, using float32
        torch.autocast(device_type="mps", dtype=torch.float32).__enter__()

    print(f"Using device: {device}")

    print("Loading GroundingDINO...")
    gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

    print("Loading SAM2 Video Predictor...")
    sam2_checkpoint = os.path.join(DATA_ROOT, "sam2_repo/sam2-main/checkpoints/sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for scene_name, prompt in PROMPTS.items():
        if test_scene and scene_name != test_scene:
            continue
            
        scene_dir = os.path.join(DATA_ROOT, scene_name)
        if not os.path.exists(scene_dir):
            print(f"Skipping {scene_name}, directory not found.")
            continue
            
        image_dir = os.path.join(scene_dir, "images")
        mask_dir = os.path.join(scene_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        image_paths = [p for p in image_paths if not os.path.basename(p).startswith("._")]
        if not image_paths:
            print(f"No images found for {scene_name}.")
            continue
            
        print(f"\nProcessing scene: {scene_name} | Prompt: '{prompt}' | Images: {len(image_paths)}")
        
        # 1. Use GroundingDINO to find the object in the FIRST frame
        first_frame_path = image_paths[0]
        image_pil = Image.open(first_frame_path).convert("RGB")
        
        print(f"Running GroundingDINO on first frame for prompt: {prompt}")
        inputs = gdino_processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gdino_model(**inputs)
            
        results = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )[0]
        
        best_box = get_largest_box(results)
        if best_box is None:
            print(f"WARNING: GroundingDINO found NO bounding boxes for '{prompt}' in {scene_name}! Skipping.")
            continue
            
        print(f"Found box: {best_box}")
        
        # 2. Initialize SAM2 Video Predictor
        print("Initializing SAM2 Video State...")
        # SAM2 requires a directory of JPEG or PNG images.
        inference_state = sam2_predictor.init_state(video_path=image_dir)
        sam2_predictor.reset_state(inference_state)
        
        # 3. Provide the bounding box to SAM2 on the first frame
        box_tensor = np.array(best_box, dtype=np.float32)
        _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box_tensor
        )
        
        # 4. Propagate through the rest of the video
        print("Propagating masks across all frames...")
        frame_names = [os.path.basename(p) for p in image_paths]
        
        with torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
                # out_mask_logits has shape (N, 1, H, W). We have 1 object.
                mask_logits = out_mask_logits[0, 0].cpu().numpy()
                mask_binary = (mask_logits > 0.0).astype(np.uint8) * 255
                
                out_filename = frame_names[out_frame_idx]
                out_path = os.path.join(mask_dir, out_filename)
                Image.fromarray(mask_binary).save(out_path)
                
                # Free memory aggressively to prevent application memory exhaustion
                if device == "mps" and out_frame_idx % 10 == 0:
                    torch.mps.empty_cache()
            
        print(f"Successfully saved {len(image_paths)} masks to {mask_dir}")

if __name__ == "__main__":
    main()
