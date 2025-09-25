import torch
import os
# os.environ['https_proxy']="http://127.0.0.1:7899"
# os.environ['TRANSFORMERS_OFFLINE']='1'
# os.environ['CUDA_VISIBLE_DEVICES']='2,3'

import pandas as pd
import torch.distributed as dist
import datasets as hf_datasets
import argparse
import transformers
from llava.model import LlavaLlamaForCausalLM
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import json
from PIL import Image
import io
from tqdm import tqdm
from llava.train.dpo_data_utils import inference_logp
from llava import conversation as conversation_lib
from llava.train.train import smart_tokenizer_and_embedding_resize
import re
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

import math

def split_dataframe(df, n):
    """Split a DataFrame into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(df) / n)
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_dataframe(lst, n)
    return chunks[k]

# def prepare_base_df(source_data_file, neg_data_file):
#     source_data = json.load(open(source_data_file))
#     neg_data = json.load(open(neg_data_file))
#
#     cap_meta_dict = {}
#     for item in source_data:
#         caption = item['conversations'][1]['value']
#         cap_meta_dict[caption] = item
#
#     columns = ['ds_name', 'text', 'origin_dataset', 'origin_split', 'idx', 'image_path']
#     ds_name = neg_data_file.split('/')[-1].replace('.json', '')
#     origin_dataset = source_data_file.split('/')[-1].replace('.json', '')
#     origin_split = json.dumps({"model": "GPT-4 anno", "type": "detailed_description"})
#     coco_dir = '../../pub_dataset/coco/train2017'
#
#     df = pd.DataFrame(columns=columns)
#     chosen_key = ''
#     rejected_key = ''
#     for i, item in enumerate(tqdm(neg_data)):
#         if chosen_key == '':
#             chosen_key = list({'prefered', 'preferred', 'chosen'}.intersection(set(item.keys())))[0]
#         if rejected_key=='':
#             rejected_key = list({'rejected', 'reject'}.intersection(set(item.keys())))[0]
#         chosen = item[chosen_key]
#         rejected = item[rejected_key].replace(' .', '.')
#
#         meta_info = cap_meta_dict[chosen]
#         question = meta_info['conversations'][0]['value'].strip()
#         image_path = os.path.join(coco_dir, meta_info['image'])
#
#         text = json.dumps({'question': question, 'chosen': chosen, 'rejected': rejected})
#
#         entry = {
#             'ds_name': ds_name,
#             # 'image': image,
#             'text': text,
#             'origin_dataset': origin_dataset,
#             'origin_split': origin_split,
#             'idx': i,
#             'image_path': image_path
#         }
#         df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
#
#     return df

def main(args):

    if args.chunk_idx == 0:
        os.makedirs(os.path.join(args.data_root, args.neg_data), exist_ok=True)
        os.makedirs(os.path.join(args.data_root, 'origin'), exist_ok=True)

    iteration = args.iteration

    # model_name = args.model_name.replace('/', '-')
    model_name = get_model_name_from_path(args.model_path)
    # source_data_file = os.path.join(args.data_root, 'source_data', args.source_data+'.json')
    # neg_data_file = os.path.join(args.data_root, 'source_data', args.neg_data+'.json')
    neg_data_parquet = os.path.join(args.data_root, args.neg_data, f'replaced_temp_{iteration}.parquet')

    model_family = get_model_name_from_path(args.model_family)
    logp_file = os.path.join(args.data_root, args.neg_data, f'{args.source_data}_{model_family}_iter_{iteration}_{args.chunk_idx}.parquet')

    # if os.path.exists(neg_data_parquet):
    #     df = pd.read_parquet(neg_data_parquet)
    # else:
    #     df = prepare_base_df(source_data_file, neg_data_file)
    df = pd.read_parquet(neg_data_parquet)
    # df = df.iloc[0:32]
    df = get_chunk(df, args.num_chunks, args.chunk_idx)

    model_base = args.model_base if 'lora' in model_name else None
    tokenizer, model, image_processor, context_len = \
        load_pretrained_model(args.model_path, model_base, model_name, device='cpu', device_map=None)

    # model = LlavaLlamaForCausalLM.from_pretrained(
    #     args.model_name,
    #     torch_dtype=torch.float16
    # )
    #
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.model_name,
    #     model_max_length=2048,
    #     padding_side="right",
    #     use_fast=False,
    # )
    #
    # if args.version == "v0":
    #     if tokenizer.pad_token is None:
    #         smart_tokenizer_and_embedding_resize(
    #             special_tokens_dict=dict(pad_token="[PAD]"),
    #             tokenizer=tokenizer,
    #             model=model,
    #         )
    # elif args.version == "v0.5":
    #     tokenizer.pad_token = tokenizer.unk_token
    # else:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     if args.version in conversation_lib.conv_templates:
    #         conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    #     else:
    #         conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    #
    # device_map=None
    # vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model(device_map=device_map)
    # if device_map != 'auto':
    #     vision_tower.to(device=device_map, dtype=torch.float16)
    # model.initialize_vision_tokenizer(args, tokenizer=tokenizer)
    # image_processor = vision_tower.image_processor

    model = model.to(dtype=torch.float16)

    device = torch.device("cuda")
    model = model.to(device)

    inference_logp(model, tokenizer, hf_datasets.Dataset.from_pandas(df),
                   logp_file,
                   image_processor, args)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Example of using argparse')

    # parser.add_argument('--model_name', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='./ckpt/llava-v1.5-7b/')
    parser.add_argument('--model_family', type=str, default='llava-v1.5-7B')

    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--freeze_backbone', type=bool, default=False)
    parser.add_argument('--tune_mm_mlp_adapter', type=bool, default=False)
    parser.add_argument('--vision_tower', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--pretrain_mm_mlp_adapter', type=str, default=None)
    parser.add_argument('--mm_use_im_patch_token', type=bool, default=False)

    # data
    parser.add_argument('--data_root', type=str, default='./playground/data/neg_data/')
    parser.add_argument('--source_data', type=str, default='detail_23k')
    parser.add_argument('--neg_data', type=str, default='detail_23k_chair_adversarial_replace_0.3_2')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument('--is_multimodal', type=bool, default=True)
    parser.add_argument('--mm_use_im_start_end', type=bool, default=False)
    parser.add_argument('--overwrite_logps', default=False, action='store_true')
    parser.add_argument('--iteration', type=int)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    # assert args.source_data in args.neg_data

    main(args)
