from models.qwen3vl_objembed import ObjectEmbed
from models.vision_process import process_vision_info
from transformers import AutoProcessor
from generate_proposal import SimpleYOLOWorldDetector

import argparse
import torch
from PIL import Image
import copy
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--objembed_checkpoint', type=str, required=True)
    parser.add_argument('--wedetect_uni_checkpoint', type=str, required=True)
    parser.add_argument('--image', nargs='+', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default=None)
    parser.add_argument('--output', type=str, default="dataset_embed.pt")

    args = parser.parse_args()

    # ----------- BỔ SUNG PHẦN NÀY -----------

    if args.image is None and args.image_folder is not None:
        image_list = []
        for file in os.listdir(args.image_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_list.append(os.path.join(args.image_folder, file))

        image_list = sorted(image_list)
        args.image = image_list

    if args.image is None:
        raise ValueError("You must provide either --image or --image_folder")

    # ----------------------------------------

    print("Embedding images:", args.image)

    # ------------------------
    # detector
    # ------------------------

    model_size = 'base' if 'base' in args.wedetect_uni_checkpoint else 'large'
    det_model = SimpleYOLOWorldDetector(
        backbone_size=model_size,
        prompt_dim=768,
        num_prompts=256,
        num_proposals=100
    )

    checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location='cpu')

    keys = list(checkpoint.keys())
    for key in keys:
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)

    keys = list(checkpoint.keys())
    for key in keys:
        if 'bbox_head' in key:
            new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
            new_key = new_key.replace('0.2.', '0.6.')
            new_key = new_key.replace('1.2.', '1.6.')
            new_key = new_key.replace('2.2.', '2.6.')
            new_key = new_key.replace('1.bn', '4')
            new_key = new_key.replace('1.conv', '3')
            new_key = new_key.replace('0.bn', '1')
            new_key = new_key.replace('0.conv', '0')
            checkpoint[new_key] = checkpoint.pop(key)

    det_model = det_model.cuda().eval()
    det_model.load_state_dict(checkpoint, strict=False)

    # ------------------------
    # objembed model
    # ------------------------

    model = ObjectEmbed.from_pretrained(
        args.objembed_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    processor = AutoProcessor.from_pretrained(args.objembed_checkpoint)

    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")

    model.model.object_token_id = object_token_index
    model = model.cuda().eval()

    candicate_object_embedding = []
    candicate_image_embedding = []
    objectnesses = []

    all_proposals = []

    for img_path in args.image:

        with torch.no_grad():
            outputs = det_model([img_path])

        proposals = [outputs[0]['bboxes'].float().cpu().tolist()]
        all_proposals.append(proposals)

        obj_str = ""
        for j in range(len(proposals[0])):
            obj_str += "Object %d: <object><object>. " % j

        image = Image.open(img_path).convert("RGB")
        ori_shape = [image.size]

        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image", "image": copy.deepcopy(image)},
                    {"type": "text", "text": obj_str}
                ]
            }
        ]

        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        texts = [processor.apply_chat_template(messages, tokenize=False).strip()]

        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )

        model_inputs = model_inputs.to(model.device)

        with torch.inference_mode():
            pred = model(
                **model_inputs,
                bboxes=[torch.tensor(proposals[0]).cuda().to(model.dtype)],
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                local_text_id=local_text_id,
            )

        candicate_object_embedding.append(pred['object_embeddings'].cpu())
        candicate_image_embedding.append(pred['full_image_embeddings'].cpu())
        objectnesses.append(pred['objness'].sigmoid().cpu())

    torch.save(
        {
            "object_embeddings": candicate_object_embedding,
            "image_embeddings": candicate_image_embedding,
            "objectness": objectnesses,
            "proposals": all_proposals,
            "images": args.image
        },
        args.output
    )

    print("Saved embedding to", args.output)