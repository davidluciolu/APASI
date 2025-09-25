'''
this is a temporary solution for no guide
'''

import copy
import os
import json
import pandas as pd
from tqdm import tqdm
import random
random.seed(42)
import re
import itertools
import glob

import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import StoppingCriteria, MinLengthLogitsProcessor, LogitsProcessorList
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path
import argparse

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import logging as transformers_logging
import math
from llava.prepare_data.merge_parquet import merge_parquet_files

transformers_logging.set_verbosity_error()

torch.multiprocessing.set_sharing_strategy('file_system')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# TODO: replace (& preprocess) captions in multiple granularity

GUIDE_TEMPLATES = [
    'There is a {}',
    'There are {}',
    '{} can also be seen',
    '{} can be seen',
    'You can see a {}',
    'There are multiple {}',
    'Several {} can be observed',
    'Some {} are present',
    'Among the items, there is a {}',
    'In the image, there is a {}',
    "On the right, there is a {}",
    "On the left, a {} is present",
    "In the center, you see a {}",
    "At the top, there is a {}",
    "At the bottom, a {} is visible",
    "In the background, a {} can be seen",
    "In the foreground, there is a {}",
    "To the side, a {} is located",
    "Near the edge, a {} appears",
    "Close to the center, a {} is seen"
]

REPLACE_TOKEN = '<REP>.'


def split_sentences_with_delimiters(text):
    pattern = r'(?<!\d)(\.)(?!\d)'

    matches = re.finditer(pattern, text)

    sentences = []
    last_index = 0

    # keep delimiters
    for match in matches:
        end_index = match.end()
        sentences.append(text[last_index:end_index].strip())
        last_index = end_index

    if last_index < len(text):
        sentences.append(text[last_index:].strip())

    return sentences

class FindReplaceCandidate:
    def __init__(self, cap_file, data_root,
                 replace_rate=0.2, sent_threshold=1, skip_sent=1, weighted_sample=False, num_neg=1,
                 use_obj_guide=False, caption_obj_file=None, save_folder=None, obj_mode='co',
                 iteration=0,
                 ):
        self.data = json.load(open(os.path.join(data_root, 'source_data', cap_file)))
        self.replace_rate = replace_rate
        self.sent_threshold = sent_threshold
        self.skip_sent = skip_sent
        self.weighted_sample = weighted_sample
        self.num_neg = num_neg
        self.use_obj_guide = use_obj_guide

        if use_obj_guide:
            caption_obj_file = os.path.join(data_root, 'source_data', caption_obj_file)
            assert os.path.exists(caption_obj_file)
            caption_obj = json.load(open(caption_obj_file))
            self.processed_captions = caption_obj['processed_captions']
            self.object_co_freq = caption_obj['node_object_co_freq']
            self.synonym_dict = caption_obj['synonym_dict']
            self.obj_mode = obj_mode

        print(f'replace rate: {self.replace_rate}, negative numbers {self.num_neg}')

        self.candidate_set = self.decide_replace()

        file_name = f'replace_candidate_{iteration}.json'
        print('replace candidate saved at: ', os.path.join(save_folder, file_name))
        json.dump(self.candidate_set, open(os.path.join(save_folder, file_name), 'w'))

    def decide_replace(self):
        candidate_set = []

        counter = 0
        for sample in tqdm(self.data):

            caption = sample['conversations'][1]['value']
            sample_id = sample['id']
            sents = split_sentences_with_delimiters(caption)
            sent_word_index = []
            start = 0
            for sent in sents:
                num_words = len(sent.split(' '))
                sent_word_index.append(list(range(start, start+num_words)))
                start = start+num_words
            sent_num = len(sents)
            replace_num = round(self.replace_rate * sent_num)

            # do not replace the first sentence
            if replace_num >= self.sent_threshold and replace_num <= (sent_num - self.skip_sent):
                for i in range(self.num_neg):
                    replace_indices = get_indexes(sent_num, replace_num, self.weighted_sample, self.skip_sent)

                    guide_objects = []
                    if self.use_obj_guide:
                        replaced_sents, guide_objects = self.get_obj_guide(sents, sample_id, sent_word_index, replace_indices)
                    else:
                        replaced_sents = [REPLACE_TOKEN if ii in replace_indices else sents[ii] for ii in range(sent_num)]

                    if replaced_sents == None:
                        continue

                    replace_candidate = ' '.join(replaced_sents)
                    replace_candidate = replace_candidate.strip()
                    if not replace_candidate.endswith('.'):
                        replace_candidate = replace_candidate+'.'

                    candidate_set.append({
                        "image_path": sample['image'],
                        "caption": caption,
                        "query": sample['conversations'][0]['value'],
                        "replace_candidate": replace_candidate,
                        "guide_objects": guide_objects,
                        "idx": counter
                    })
                    counter += 1

        return candidate_set

    def get_obj_guide(self, sents, sample_id, sent_word_index, replace_indices):

        processed_caption = self.processed_captions[sample_id]
        replaced_sents = copy.deepcopy(sents)

        remaining_word_index = [sent_word_index[j] for j in range(len(sents)) if j not in replace_indices]
        remaining_word_index_flatten = list(itertools.chain(*remaining_word_index))
        remaining_objects = []
        for obj, ind in zip(processed_caption['node_objects'], processed_caption['indexes']):
            if ind[0] in remaining_word_index_flatten:
                remaining_objects.append(obj)
        if len(remaining_objects) == 0:
            return None, None

        no_exist_node_objects = set(self.object_co_freq.keys())-set(processed_caption['node_objects'])
        guide_objects = []
        for rep_id in replace_indices:
            template = random.choice(GUIDE_TEMPLATES)
            key_object = random.choice(remaining_objects)

            if self.obj_mode == 'co':
                co_objs = []
                for co_obj in self.object_co_freq[key_object].keys():
                    if co_obj in no_exist_node_objects:
                        co_objs.append(co_obj)
                    if len(co_objs) >= 10:
                        break
                if len(co_objs) == 0:
                    guide_node_object = random.choice(list(self.object_co_freq.keys()))
                else:
                    guide_node_object = random.choice(co_objs)
            else:
                guide_node_object = random.choice(list(no_exist_node_objects))
            guide_object = random.choice(self.synonym_dict[guide_node_object])

            replaced_sents[rep_id] = template.format(guide_object) + ' ' + REPLACE_TOKEN
            guide_objects.append(guide_object)

        return replaced_sents, guide_objects


class InferenceDataset(Dataset):
    def __init__(self, candidate_list):
        self.candidate_set = candidate_list
        self.index_mapping = {item['idx']:i for i, item in enumerate(self.candidate_set)}

    def index_global_to_local(self, global_index):
        return self.index_mapping[global_index]

    def __getitem__(self, index):
        return self.candidate_set[index]

    def __len__(self):
        return len(self.candidate_set)


def get_indexes(n, m, weighted, skip_sent):
    numbers = torch.arange(skip_sent, n)
    if weighted and n>(skip_sent+1):
        a = 1 / (torch.sqrt(torch.tensor(n - 1)) - torch.sqrt(torch.tensor(skip_sent)))
        b = 1 - torch.sqrt(torch.tensor(skip_sent))*a
        weights = a * torch.sqrt(numbers) + b
    else:
        weights = [1] * (n - skip_sent)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, num_samples=m, replacement=False)

    selected = numbers[indices].tolist()

    return selected


def replace_sentence_wrap(
    model, tokenizer,
    cap_file, data_root, folder_name,
    image_dir='./playground/data/coco/train2017', conv_mode='vicuna_v1',
    iteration=0
):

    ds_name = folder_name.split('/')[-1]
    origin_dataset = cap_file.split('/')[-1].replace('.json', '')
    origin_split = json.dumps({"model": "GPT-4 anno", "type": "detailed_description"})

    candidate_file = os.path.join(folder_name, f'replace_candidate_{iteration}.json')
    candidate_list = json.load(open(candidate_file))
    candidate_list = get_chunk(candidate_list, args.num_chunks, args.chunk_idx)

    d_set = InferenceDataset(candidate_list=candidate_list)
    # dd = dataset[0]
    dataloader = DataLoader(dataset=d_set, batch_size=1, num_workers=2, shuffle=False)

    zip_outputs = replace_sentence(
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        conv_mode=conv_mode,
        )

    reject_sentences, origin_indices = zip_outputs

    columns = ['ds_name', 'text', 'origin_dataset', 'origin_split', 'idx', 'image_path']
    df = pd.DataFrame(columns=columns)
    for i in range(len(reject_sentences)):
        rej_sent = reject_sentences[i]
        org_ind = d_set.index_global_to_local(origin_indices[i].item())
        if rej_sent != '':
            origin_data = d_set[org_ind]
            text = json.dumps({'question': origin_data["query"],
                               'chosen': origin_data["caption"],
                               'rejected': rej_sent})
            entry = {
                'ds_name': ds_name,
                # 'image': image,
                'text': text,
                'origin_dataset': origin_dataset,
                'origin_split': origin_split,
                'idx': origin_data["idx"],
                'image_path': os.path.join(image_dir, origin_data['image_path']),
                'guide_object': origin_data['guide_objects']
            }

            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    file_name = os.path.join(folder_name, f'replaced_temp_{iteration}_{args.chunk_idx}.parquet')

    df.to_parquet(file_name)
    print(f'orginal sample: {len(d_set)}, collected: {len(df)}')
    print(f'save to {file_name}')


def replace_sentence(dataloader, model, tokenizer, conv_mode,
                     ):
    outputs = []
    indices = []

    # TODO: indentify replace token, and replace
    for batch in tqdm(dataloader):
        batch_len = len(batch["caption"])
        for idx in range(batch_len):

            replace_candidate = batch["replace_candidate"][idx]
            query = batch["query"][idx]
            index = batch["idx"][idx]
            context_parts = [part.strip() for part in replace_candidate.split(REPLACE_TOKEN)]
            if context_parts[-1] == '':
                context_parts.pop()

            replaced_sents = ''
            for part in context_parts:
                if replaced_sents == '':
                    replaced_sents = part
                else:
                    if part == '':
                        replaced_sents = replaced_sents + part
                    else:
                        replaced_sents = replaced_sents + ' ' + part
                    if part.endswith('.'):  # last sentence, no replace
                        continue
                replaced_sents = replaced_sents + ' '
                context = get_context(query, replaced_sents, conv_mode)
                replaced_res = unimodal_generate(context, model, tokenizer, conv_mode)
                replaced_sents = replaced_sents + replaced_res
                replaced_sents = replaced_sents.strip()
                if not replaced_sents.endswith('.'):
                    replaced_sents = replaced_sents + '.'

            outputs.append(replaced_sents)
            indices.append(index)
        #
        # if len(outputs)==3:
        #     break

    zip_outputs = (outputs, indices)

    return zip_outputs


def get_context(query, answer, conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], query.replace('<image>', '').strip())
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if answer == '':
        prompt = conv.get_prompt()
        prompt = prompt + conv.roles[1] + ': '                  # add Assistant:
    else:
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        prompt = prompt[:]
        prompt = re.sub(re.escape(stop_str)+'$', '', prompt)    # remove conv sep at end
        # prompt = re.sub(re.escape('. ') + '$', '', prompt)
    return prompt.strip()


@torch.inference_mode()
def unimodal_generate(context, model, tokenizer, conv_mode):
    torch.cuda.empty_cache()

    input_ids = tokenizer(context)['input_ids']
    input_ids = torch.as_tensor(input_ids).reshape(1, -1)
    input_size = input_ids.shape[-1]

    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria =KeywordsStoppingCriteria(keywords=[stop_str, '.'],
                                                tokenizer=tokenizer,
                                                input_ids=input_ids,
                                                )

    logit_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=tokenizer(stop_str).input_ids)])

    output = model.generate(
        input_ids.cuda(),
        images=None,
        num_beams=1,
        do_sample=False,
        temperature=0,
        max_new_tokens=64,
        stopping_criteria=[stopping_criteria],
        logits_processor=logit_processor,
        # early_stopping=True,
        repetition_penalty=1.1)

    # response = self.tokenizer.batch_decode(output[:, input_size:], skip_special_tokens=True)[0]   # why input_size: ?
    response = tokenizer.batch_decode(output[:, :], skip_special_tokens=True)[0]
    if response.count(stop_str):
        response = response[: response.index(stop_str)]
    if response.count('.'):
        response = response[: response.index('.')]
    return response.strip()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--model_family', type=str, default='llava-v1.5-7b')
    parser.add_argument('--cap_file', type=str, default='detail_23k.json')
    parser.add_argument('--data_root', type=str, default='./playground/data/neg_data/')
    parser.add_argument('--image_dir', type=str, default='./playground/data/coco/train2017')
    parser.add_argument('--replace_rate', type=float, default=0.2, help='Replace rate')
    parser.add_argument('--sent_threshold', type=int, default=1, help='Sentence threshold')
    parser.add_argument('--skip_sent', type=int, default=1, help='Skip sentence')
    parser.add_argument('--weighted_sample', action='store_true', help='Weighted sample')
    parser.add_argument('--use_obj_guide', action='store_true', help='object guided llm replace')
    parser.add_argument('--caption_obj_file', type=str, help='files with object info of captions', default='detail_23k_chair_processed.json')
    parser.add_argument('--num_neg', type=int, help='negative per sample', default=5)
    parser.add_argument('--run_id', type=str, default=None, help='Rep ID')
    parser.add_argument('--obj_mode', type=str, default='co', help='guide object mode, co-occur, freq, random ', choices=['coo', 'freq', 'rand'])
    parser.add_argument('--find_rep_candidate', action='store_true', help='object guided llm replace')
    parser.add_argument('--iteration', type=int)

    parser.add_argument('--conv-mode', type=str, default='vicuna_v1')

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--merge-parquets", action='store_true', help='object guided llm replace')

    args = parser.parse_args()

    if ('llava-v1.5' in args.model_path) or ('vicuna' in args.model_path) or ('RLAIF' in args.model_path):
        args.conv_mode = 'vicuna_v1'

    elif ('mistral' in args.model_path):
        args.conv_mode = 'mistral_direct'

    model_name = get_model_name_from_path(args.model_path)
    model_family = get_model_name_from_path(args.model_family)

    folder_name = args.cap_file.replace('.json', '')

    guide_name = 'no'
    if args.use_obj_guide:
        if 'chair' in args.caption_obj_file.lower():
            guide_name = 'chair'
        else:
            guide_name = 'lvis'

    if args.run_id is None:
        folder_name = f'{folder_name}_{model_family}_{guide_name}_guide_replace_{args.replace_rate}_{args.sent_threshold}_skip{args.skip_sent}_num{args.num_neg}'
    else:
        folder_name = f'{folder_name}_{model_family}_{guide_name}_guide_replace_{args.replace_rate}_{args.sent_threshold}_skip{args.skip_sent}_num{args.num_neg}_{args.run_id}'
    folder_name = os.path.join(args.data_root, folder_name)

    if args.find_rep_candidate:
        os.makedirs(folder_name, exist_ok=True)

        findrep = FindReplaceCandidate(args.cap_file, args.data_root,
                                       replace_rate=args.replace_rate, sent_threshold=args.sent_threshold,
                                       skip_sent=args.skip_sent, weighted_sample=args.weighted_sample, num_neg=args.num_neg,
                                       use_obj_guide=args.use_obj_guide, caption_obj_file=args.caption_obj_file,
                                       save_folder=folder_name, obj_mode=args.obj_mode,
                                       iteration=args.iteration
                                       )

    elif args.merge_parquets:
        prefix = os.path.join(folder_name, f'replaced_temp_{args.iteration}')
        merge_parquet_files(prefix, args.num_chunks)
        temp_files = glob.glob(f'{prefix}_*.parquet')
        for file in temp_files:
            os.remove(file)

    else:
        model_base = args.model_base

        tokenizer, model, _, _ = load_pretrained_model(args.model_path, model_base, model_name, device='cpu', device_map=None)
        device = torch.device(f"cuda")
        model = model.to(device)

        replace_sentence_wrap(model=model, tokenizer=tokenizer, cap_file=args.cap_file,
                                  data_root=args.data_root, folder_name=folder_name, image_dir=args.image_dir,
                                  iteration=args.iteration, conv_mode=args.conv_mode)
