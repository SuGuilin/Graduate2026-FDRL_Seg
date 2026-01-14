import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Force specific GPU if needed via env var before importing torch (optional)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
import cv2
from PIL import Image as PILImage
import json
import re

# === Fusion model import (from FMTS_Fusion) ===
try:
    from FMTS_Fusion.network import SGLMamba as FusionNet
except Exception as e:
    raise ImportError("Cannot import fusion network. Make sure FMTS-Fusion folder is on PYTHONPATH and network_WMamba.py exists.") from e

# === Reasoning + processor + helper (from InsReasoner) ===
try:
    from transformers import AutoProcessor
    # For loading model we will import inside function (to avoid heavy imports if not used)
    from qwen_vl_utils import process_vision_info
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    # If imports fail, provide clearer message
    print("Warning: Could not import InsReasoner utilities (transformers/qwen_vl_utils/sam2).")
    print("Make sure InsReasoner folder is on PYTHONPATH and dependencies are installed.")
    # continue; imports will be retried later where needed

# -------------------------
# Utilities
# -------------------------
def read_image_cv2_rgb(path):
    """Read with cv2 as color (BGR) -> convert to RGB, normalize to [0,1], return torch tensor [1,3,H,W] float32"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    # get original height and width
    h, w = img.shape[:2]
    new_h = (h // 4) * 4
    new_w = (w // 4) * 4

    # resize to (h//4*4, w//4*4) if necessary
    if h % 4 or w % 4:
        img = cv2.resize(img, (new_w, new_h))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW
    tensor = torch.from_numpy(img).unsqueeze(0).float()  # [1,3,H,W]
    return tensor, (h, w)

def fused_tensor_to_pil(img_tensor, original_size):
    """
    Convert tensor [1,3,H,W] or [3,H,W] in float (assume in [0,1] or slightly out) to PIL RGB uint8
    """
    if isinstance(img_tensor, torch.Tensor):
        arr = img_tensor.detach().cpu().numpy()
    else:
        arr = np.array(img_tensor)
    if arr.ndim == 4:
        arr = arr[0]
    # CHW -> HWC
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    arr = arr.transpose(1, 2, 0)

    # resize to original size
    original_h, original_w = original_size
    if original_h % 4 or original_w % 4:
        arr = cv2.resize(arr, (original_w, original_h))
    return PILImage.fromarray(arr)  # RGB

# -------------------------
# Fusion loader & run
# -------------------------
def load_fusion_model(model_path, in_channel=3, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"[fusion] device: {device}")
    model = FusionNet(in_chans=in_channel, embed_dim=192).to(device)
    ckpt = torch.load(model_path, map_location=device)
    param_key = "params"
    state_dict = ckpt[param_key] if param_key in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device

@torch.no_grad()
def run_fusion_once(model, device, vi_path, ir_path):
    """
    Read vi + ir -> tensors -> forward -> return fused tensor [1,3,H,W] float in [0,1]
    Assumes model takes (vi, ir) as inputs like earlier.
    """
    img_vi, original_size = read_image_cv2_rgb(vi_path)  # [1,3,H,W]
    img_vi = img_vi.to(device)
    img_ir, _ = read_image_cv2_rgb(ir_path) # we read IR as 3-channel (as you requested)
    img_ir = img_ir.to(device) 
    # If your IR is single-channel, you can adapt here to replicate channel or convert

    # If model expects different ordering (ir, vi) adjust call accordingly. You used model(img_vi, img_ir) above.
    out = model(img_vi, img_ir)  # expect torch tensor [1,3,H,W] or similar
    # ensure in float and normalized: try to map to [0,1] if output range unknown
    out = out.detach().cpu().float()
    # If model outputs in 0-255, convert. We assume output in [0,1] (consistent with previous demo).
    # But to be safe, clamp and if max>1 treat as 0-255 scale:
    if out.max() > 1.1:
        # assume 0-255
        out = out / 255.0
    out = torch.clamp(out, 0.0, 1.0)
    return out, original_size  # [1,3,H,W]

# -------------------------
# Reasoning + SAM pipeline
# -------------------------
def load_reasoning_and_segmentation(reasoning_model_path, segmentation_model_path, device):
    # lazy imports for heavy libs
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    # choose model class: use Qwen2_5_VLForConditionalGeneration if available else generic loader by from_pretrained
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModelClass
    except Exception:
        # fallback to generic import for large models; user should ensure correct class is importable
        try:
            from qwen_vl_model import Qwen2_5_VLForConditionalGeneration as QwenModelClass
        except Exception:
            QwenModelClass = None

    if QwenModelClass is None:
        # We will load with from_pretrained dynamically (the original code used Qwen2_5_VLForConditionalGeneration.from_pretrained)
        from transformers import AutoModelForCausalLM
        print("Warning: Qwen model class not found as named import; attempting generic AutoModelForCausalLM (may fail).")

    # load processor
    processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
    # load reasoning model (we prefer Qwen2_5_VLForConditionalGeneration if available)
    if QwenModelClass is not None:
        reasoning_model = QwenModelClass.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        reasoning_model = None
    # load SAM2 predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

    if reasoning_model is not None:
        reasoning_model.eval()

    return processor, reasoning_model, segmentation_model

def extract_bbox_points_think(output_text, x_factor, y_factor):
    """
    Parse output_text produced by the Qwen template to extract JSON inside <answer>...</answer> and optional <think>.
    Returns lists of bboxes and points and think_text.
    """
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    pred_bboxes, pred_points = [], []
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            pred_bboxes = [
                [
                    int(item['bbox_2d'][0] * x_factor + 0.5),
                    int(item['bbox_2d'][1] * y_factor + 0.5),
                    int(item['bbox_2d'][2] * x_factor + 0.5),
                    int(item['bbox_2d'][3] * y_factor + 0.5)
                ] for item in data if len(item.get('bbox_2d', [])) == 4
            ]
            pred_points = [
                [
                    int(item['point_2d'][0] * x_factor + 0.5),
                    int(item['point_2d'][1] * y_factor + 0.5)
                ] for item in data if len(item.get('bbox_2d', [])) == 4
            ]
        except Exception as e:
            print("Failed to parse JSON in <answer>: ", e)
    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    return pred_bboxes, pred_points, think_text

@torch.no_grad()
def run_reasoning_and_sam(processor, reasoning_model, segmentation_model, fused_pil_image, instruction, output_path, device, think_output_path):
    """
    fused_pil_image: PIL.Image RGB
    processor, reasoning_model: loaded objects (reasoning_model may be None for this simplified pipeline)
    segmentation_model: SAM2ImagePredictor
    instruection: text string for the Qwen template
    Saves overlay image at output_path (and also 'raw_mask.npy' alongside)
    """
    # Prepare prompt & message template (same as user's earlier code)
    QUESTION_TEMPLATE = (
        "Please find \"{Question}\" with bboxs and points. "
        "Compare the difference between object(s) and find the most closely matched object(s). "
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )

    # Resize parameters same as your previous code
    original_width, original_height = fused_pil_image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    # Build the messages with the fused image
    image_for_model = fused_pil_image.resize((resize_size, resize_size), PILImage.BILINEAR)

    # Apply processor template
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image_for_model},
            {"type": "text",
             "text": QUESTION_TEMPLATE.format(
                 Question=instruction.lower().strip("."),
                 # Provide dummy Answer template - the model will generate real JSON, but template helps formatting
                 Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}]'
             )
            }
        ]
    }]]

    # Convert messages -> processor text and image tensors
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate with reasoning model
    if reasoning_model is None:
        # If reasoning model not loaded, we will skip generation and use the template's placeholder answer.
        print("[reasoning] reasoning model not available — using placeholder answer. SAM will use placeholder bbox/point.")
        output_text = messages[0][0]['content'][1]['text']  # the template text that includes placeholder <answer>
    else:
        # Use autocast to save memory if possible
        gen_kwargs = dict(use_cache=True, max_new_tokens=1024, do_sample=False)
        with torch.inference_mode(), torch.autocast(device.type if device.type!='cpu' else 'cpu', dtype=torch.bfloat16 if device.type!='cuda' else torch.bfloat16):
            generated_ids = reasoning_model.generate(**inputs, **gen_kwargs)
        # trim prefix (same logic as earlier)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_text = output_texts[0]

    print("[reasoning] model output text:")
    print(output_text)
    with open(think_output_path, "w") as f:
        f.write(output_text)

    # Parse bboxes/points
    bboxes, points, think_text = extract_bbox_points_think(output_text, x_factor, y_factor)
    print(f"[reasoning] extracted {len(bboxes)} boxes and {len(points)} points. Think text length: {len(think_text)}")

    # If no bboxes extracted, fallback to center point
    if len(points) == 0:
        cx, cy = original_width // 2, original_height // 2
        points = [[cx, cy]]
        bboxes = [[max(0, cx - 50), max(0, cy - 50), min(original_width, cx + 50), min(original_height, cy + 50)]]
        print("[reasoning] fallback: using image center as point and bbox.")

    # SAM2 segmentation
    segmentation_model.set_image(fused_pil_image)
    mask_all = np.zeros((fused_pil_image.height, fused_pil_image.width), dtype=bool)
    for bbox, point in zip(bboxes, points):
        masks, scores, _ = segmentation_model.predict(point_coords=[point], point_labels=[1], box=bbox)
        if len(scores) == 0:
            continue
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        mask = masks[0].astype(bool)
        mask_all = np.logical_or(mask_all, mask)

    # Save raw mask
    #mask_path = os.path.splitext(output_path)[0] + "_mask.npy"
    #np.save(mask_path, mask_all)
    #print(f"[sam] saved raw mask -> {mask_path}")

    # Create RGBA overlay with semi-transparent red mask
    image_rgba = fused_pil_image.convert("RGBA")
    overlay = PILImage.new("RGBA", fused_pil_image.size, (0,0,0,0))
    mask_arr = np.zeros((fused_pil_image.height, fused_pil_image.width, 4), dtype=np.uint8)
    mask_arr[mask_all] = [255, 0, 0, 80]  # red with alpha 80
    mask_overlay_pil = PILImage.fromarray(mask_arr, mode='RGBA')
    final = PILImage.alpha_composite(image_rgba, mask_overlay_pil)
    final.save(output_path)
    print(f"[sam] saved overlay result -> {output_path}")

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end: FMFS-Fusion Fuse VI+IR -> InsReasoner + SAM2")
    parser.add_argument("--fusion_model_path", type=str, required=True, help="Path to fusion checkpoint (.pth)")
    parser.add_argument("--reasoning_model_path", type=str, required=True, help="Path or huggingface id for reasoning model")
    parser.add_argument("--segmentation_model_path", type=str, required=False, default="facebook/sam2-hiera-large", help="SAM2 model id/path")
    parser.add_argument("--vi_path", type=str, required=True, help="Visible RGB image path")
    parser.add_argument("--ir_path", type=str, required=True, help="Infrared image path (read as 3-channel BGR -> RGB)")
    parser.add_argument("--instruction", type=str, required=True, default="Find the bicycle and return its bbox and point.", help="Query text for reasoning model")
    parser.add_argument("--output_path", type=str, required=True, help="Final overlay output path")
    parser.add_argument("--think_output_path", type=str, required=True, help="Final thinking process output path")
    parser.add_argument("--tmp_fused_path", type=str, required=False, default="./out_fused.png", help="Optional: save intermediate fused image path")
    parser.add_argument("--in_channel", type=int, default=3, help="Input channels to fusion model (default 3)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.tmp_fused_path:
        os.makedirs(os.path.dirname(args.tmp_fused_path) or ".", exist_ok=True)

    # Device for fusion (use same device for reasoning if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load fusion model
    print("Loading fusion model...")
    fusion_model, fusion_device = load_fusion_model(args.fusion_model_path, in_channel=args.in_channel, device=device)

    # Run fusion
    print("Running fusion on input images...")
    fused_tensor, original_size = run_fusion_once(fusion_model, fusion_device, args.vi_path, args.ir_path)  # [1,3,H,W] float in [0,1]

    # Convert fused tensor to PIL RGB
    fused_pil = fused_tensor_to_pil(fused_tensor, original_size)

    # Optionally save fused image
    if args.tmp_fused_path:
        fused_pil.save(args.tmp_fused_path)
        print(f"[fusion] saved intermediate fused image -> {args.tmp_fused_path}")

    # Load reasoning + segmentation models (this may be heavy)
    print("Loading reasoning and segmentation models (this may take a while)...")
    processor, reasoning_model, segmentation_model = load_reasoning_and_segmentation(args.reasoning_model_path, args.segmentation_model_path, device)

    # Run reasoning + SAM and save overlay
    print("Running reasoning + SAM2 segmentation...")
    run_reasoning_and_sam(processor, reasoning_model, segmentation_model, fused_pil, args.instruction, args.output_path, device, args.think_output_path)
    print("Done.")

if __name__ == "__main__":
    main()
