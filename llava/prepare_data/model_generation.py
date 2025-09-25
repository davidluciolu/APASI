'''
a script for open-ended VQA,
write GT answers and meta info in the answer file
'''

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["conversations"][0]['value']

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # qs = qs.replace(DEFAULT_IMAGE_TOKEN, '')

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size, index

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, question_indices = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = (input_ids_padded != IGNORE_INDEX).long()  # 生成 attention mask


    image_tensors = torch.stack(image_tensors, dim=0)

    return input_ids_padded, attention_mask, image_tensors, image_sizes, question_indices


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = args.model_base if 'lora' in model_path.lower() else None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # avoid empty output when batching generation
    tokenizer.padding_side = "left"
    model.config.tokenizer_padding_side = 'left'

    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans = []

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config,
                                     batch_size=args.batch_size)

    # for line in tqdm(questions):
    #     image_file = line["image"]
    #     qs = line["conversations"][0]['value']
    #
    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    #
    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    #
    #     image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
    #     image_tensor = process_images([image], image_processor, model.config)[0]
    #
    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=image_tensor.unsqueeze(0).half().cuda(),
    #             image_sizes=[image.size],
    #             do_sample=True if args.temperature > 0 else False,
    #             temperature=args.temperature,
    #             top_p=args.top_p,
    #             num_beams=args.num_beams,
    #             # no_repeat_ngram_size=3,
    #             max_new_tokens=1024,
    #             use_cache=True)
    #
    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #     new_line = deepcopy(line)
    #     new_line["conversations"][1]['value'] = outputs
    #
    #     ans.append(new_line)

    skip = -1
    temp_answers_file = answers_file.replace('.json', '_temp.json')
    if os.path.exists(temp_answers_file):
        print(f'load saved temp at: {temp_answers_file}')
        temp_file = json.load(open(temp_answers_file))
        skip = temp_file['skip']
        ans = temp_file['ans']
        # os.remove(temp_answers_file)

    for ii, (input_ids, attention_mask, image_tensor, image_sizes, question_indices) in enumerate(tqdm(data_loader)):
        torch.cuda.empty_cache()

        if ii <= skip:
            continue

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = attention_mask.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            # (
            #     inputs,
            #     position_ids,
            #     attention_mask,
            #     _,
            #     inputs_embeds,
            #     _
            # ) = model.prepare_inputs_labels_for_multimodal(
            #     input_ids,
            #     None,
            #     attention_mask,
            #     None,
            #     None,
            #     images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            #     image_sizes=image_sizes
            # )
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # print(outputs)
            # assert 1==0

        for question_index, output in zip(question_indices, outputs):
            output = output.strip()
            new_line = deepcopy(questions[question_index])
            new_line["conversations"][1]['value'] = output
            ans.append(new_line)

        if ii > 0 and (ii % 2000 == 0):
            json.dump({'skip': ii, 'ans': ans},
                      open(temp_answers_file, 'w'), indent=4)

        # if len(ans) >= 10:
        #     break

    json.dump(ans, open(answers_file, 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if ('llava-v1.5' in args.model_path) or ('vicuna' in args.model_path) or ('RLAIF' in args.model_path):
        args.conv_mode = 'vicuna_v1'

    # elif ('mistral' in args.model_path):
    #     args.conv_mode = 'mistral_direct'

    eval_model(args)
