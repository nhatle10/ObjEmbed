from models.qwen3vl_objembed import ObjectEmbed
from models.vision_process import process_vision_info
from transformers import AutoProcessor
from generate_proposal import SimpleYOLOWorldDetector
from vis import plot_bounding_boxes
from visualize import plot_topk
import argparse
import torch
from PIL import Image
import copy
import torch.nn.functional as F


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--objembed_checkpoint', type=str, required=True)
    parser.add_argument('--wedetect_uni_checkpoint', type=str, default='')

    parser.add_argument('--image', nargs='+', type=str, default=None)
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--image_query', type=str, default='')

    parser.add_argument('--task', type=str, default='rec',
                        choices=['rec', 'retrieval_by_object', 'retrieval_by_image'])

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument('--embedding_file', type=str, default=None)

    args = parser.parse_args()

    print('Input image:', args.image)
    print("query:", args.query)

    if args.task == 'rec':
        assert len(args.image) == 1

    # ====================================
    # LOAD PRECOMPUTED EMBEDDINGS
    # ====================================

    use_precomputed = False

    if args.embedding_file is not None:

        data = torch.load(args.embedding_file)

        candicate_object_embedding = torch.cat(data["object_embeddings"], dim=0).cuda()
        candicate_image_embedding = torch.cat(data["image_embeddings"], dim=0).cuda()
        objectnesses = torch.cat(data["objectness"], dim=0).cuda()

        proposals = data["proposals"]
        args.image = data["images"]

        use_precomputed = True

        print("Loaded embeddings from:", args.embedding_file)

    # ====================================
    # LOAD DETECTOR
    # ====================================

    if not use_precomputed:

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

    # ====================================
    # LOAD OBJEMBED
    # ====================================

    model_kwargs = dict(
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    model = ObjectEmbed.from_pretrained(args.objembed_checkpoint, **model_kwargs)

    processor = AutoProcessor.from_pretrained(args.objembed_checkpoint)

    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")

    model.model.object_token_id = object_token_index

    global_id = None
    global_text_id = None

    if model.use_global_caption:
        global_id = processor.tokenizer.convert_tokens_to_ids("<global>")
        global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>")

    model = model.cuda().eval()

    # ====================================
    # QUERY EMBEDDING
    # ====================================

    if args.query != '':

        if args.task in ['rec', 'retrieval_by_object']:

            text_prompt = f"Find an object that matches the given caption. {args.query} <local_text>"

        else:

            text_prompt = f"Find an image that matches the given caption. {args.query} <global_text>"

        messages = [{"role": "user", "content":[{"type":"text","text":text_prompt}]}]

        texts = [processor.apply_chat_template(messages, tokenize=False).strip()]

        model_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        ).to(model.device)

        with torch.inference_mode():

            pred = model(
                text_processed=model_inputs,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id
            )

        if args.task in ['rec','retrieval_by_object']:
            query_embedding = pred['local_text_embeddings']
        else:
            query_embedding = pred['global_text_embeddings']

    else:

        image = Image.open(args.image_query).convert("RGB")

        ori_shape = [image.size]

        messages = [{
            "role":"user",
            "content":[
                {"type":"image","image":copy.deepcopy(image)},
                {"type":"text","text":"The coarse global image is <global>. The detailed global image is <global>."}
            ]
        }]

        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        texts = [processor.apply_chat_template(messages, tokenize=False).strip()]

        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        ).to(model.device)

        with torch.inference_mode():

            pred = model(
                **model_inputs,
                bboxes=[torch.zeros((0,4)).cuda().to(model.dtype)],
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id,
            )

        query_embedding = pred['full_image_embeddings'][:,0,:]

    # ====================================
    # RETRIEVAL BY OBJECT
    # ====================================

    if args.task == 'retrieval_by_object':

        candicate_object_embedding = F.normalize(candicate_object_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)

        pred_scores = query_embedding @ candicate_object_embedding.transpose(-1,-2)

        pred_scores = pred_scores * model.logit_log_scale.exp()
        pred_scores = pred_scores + model.logit_bias

        pred_scores = pred_scores.float().sigmoid().cpu()

        pred_scores = pred_scores.reshape(len(args.image),100)

        # image_scores = torch.max(pred_scores, dim=1)[0]

        # topk = min(args.topk, len(image_scores))

        # scores, idxs = torch.topk(image_scores, topk)

        # print("\nTop-K images:")

        # topk_images = []

        # for rank,(i,s) in enumerate(zip(idxs,scores)):
        #     print(f"{rank+1}. {args.image[i]}  score={s:.4f}")
        #     topk_images.append(args.image[i])

        # if args.visualize:
        #     plot_topk(topk_images, scores.tolist())

        image_scores, best_obj_idx = torch.max(pred_scores, dim=1)

        topk = min(args.topk, len(image_scores))
        scores, idxs = torch.topk(image_scores, topk)

        topk_images = []
        topk_boxes = []

        print("\nTop-K images:")

        for rank,(i,s) in enumerate(zip(idxs,scores)):

            img_path = args.image[i]
            obj_id = best_obj_idx[i]

            bbox = proposals[i][0][obj_id]

            print(f"{rank+1}. {img_path} score={s:.4f}")

            topk_images.append(img_path)
            topk_boxes.append(bbox)

        if args.visualize:
            plot_topk(topk_images, scores.tolist())

    # ====================================
    # RETRIEVAL BY IMAGE
    # ====================================

    elif args.task == 'retrieval_by_image':

        candicate_image_embedding = F.normalize(candicate_image_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)

        pred_scores = query_embedding @ candicate_image_embedding.transpose(-1,-2)

        pred_scores = pred_scores * model.logit_image_log_scale.exp()
        pred_scores = pred_scores + model.logit_image_bias

        pred_scores = pred_scores.float().sigmoid().cpu().flatten()

        topk = min(args.topk, len(pred_scores))

        scores, idxs = torch.topk(pred_scores, topk)

        print("\nTop-K images:")

        topk_images = []

        for rank,(i,s) in enumerate(zip(idxs,scores)):
            print(f"{rank+1}. {args.image[i]}  score={s:.4f}")
            topk_images.append(args.image[i])

        if args.visualize:
            plot_topk(topk_images, scores.tolist())

    # ====================================
    # REC
    # ====================================

    elif args.task == 'rec':

        candicate_object_embedding = F.normalize(candicate_object_embedding, dim=-1)
        query_embedding = F.normalize(query_embedding, dim=-1)

        pred_scores = query_embedding @ candicate_object_embedding.transpose(-1,-2)

        pred_scores = pred_scores * model.logit_log_scale.exp()
        pred_scores = pred_scores + model.logit_bias

        pred_scores = pred_scores.float().sigmoid().cpu().flatten()

        pred_scores = pred_scores * objectnesses.cpu().flatten()

        if args.visualize:

            max_index = torch.argmax(pred_scores)

            image = Image.open(args.image[0]).convert("RGB")

            pred_image = plot_bounding_boxes(image, [proposals[0][max_index]])

            pred_image.save("pred.png")

            print("Saved result → pred.png")