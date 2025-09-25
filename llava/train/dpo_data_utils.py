import io
import json
import os

import tqdm
import copy
import torch
import itertools
import pandas as pd
import torch.utils.data as torch_data
from PIL import Image
from typing import Dict, Optional, Sequence, List
import math

from llava.train.train import preprocess, preprocess_multimodal, DataCollatorForSupervisedDataset
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,\
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib
from functools import partial

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



def encode_multimodal_preference_sample(source,
                                        tokenizer,
                                        image_processor,
                                        args):
    processor = image_processor
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    has_image = 'image' in source
    if has_image:
        image = source['image']
        if args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        win_conv = preprocess_multimodal([win_conv], args)
        rej_conv = preprocess_multimodal([rej_conv], args)


    win_data_dict = preprocess(win_conv, tokenizer, has_image=has_image)
    win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                         labels=win_data_dict["labels"][0])

    rej_data_dict = preprocess(rej_conv, tokenizer, has_image=has_image)
    rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                         labels=rej_data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image

    if 'ref_win_logp' in source:
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
    return rej_data_dict, win_data_dict


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 image_processor,
                 args,):

        self.data = data

        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": json.loads(sample['origin_split']),
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        text = json.loads(sample['text'])
        question = {'from': 'human', 'value': text['question']}
        chosen = {'from': 'gpt', 'value': text['chosen']}
        rejected = {'from': 'gpt', 'value': text['rejected']}

        image = Image.open(sample['image_path']).convert('RGB')

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(formated_sample,
                                                                           self.tokenizer,
                                                                           self.image_processor,
                                                                           self.args)
        return rej_data_dict, win_data_dict, \
               # sample['idx']

    def __len__(self):
        return len(self.data)


class RLHFVDataset(torch_data.Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.data = pd.read_parquet(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        text = json.loads(sample['text'])
        question = {'from': 'human', 'value': f"{text['question']}"}
        chosen = {'from': 'gpt', 'value': text['chosen']}
        rejected = {'from': 'gpt', 'value': text['rejected']}

        image = 'place_holder'

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }


        (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
         data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp'],) \
            = text['logps']

        return data_dict

def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def preference_collator_fn(instances, tokenizer):

    # rej_instances, win_instances, index = list(zip(*instances))
    rej_instances, win_instances = list(zip(*instances))
    # TODO: check length of ids and lopgs

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    rej_batch = data_collator(rej_instances)
    win_batch = data_collator(win_instances)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], tokenizer.pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], IGNORE_INDEX)
    concatenated_attention_mask = concatenated_input_ids.ne(tokenizer.pad_token_id)

    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'],
        # index=index,
    )
    return batch

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of IGNORE_INDEX(IGNORE_INDEX) are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != IGNORE_INDEX)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == IGNORE_INDEX] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob


def get_multimodal_sample_logps(model, dataloader):
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    # index_list = []
    # win_input_ids_list = []
    # rej_input_ids_list = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):

            win_input_ids = batch.pop('win_input_ids')
            rej_input_ids = batch.pop('rej_input_ids')

            images = batch.pop('images')
            concatenated_input_ids = batch.pop('concatenated_input_ids')
            concatenated_labels = batch.pop('concatenated_labels')
            concatenated_attention_mask = batch.pop('concatenated_attention_mask')
            concatenated_images = torch.cat([images, images], dim=0)

            output = model(
                input_ids=concatenated_input_ids.cuda(),
                labels=concatenated_labels.cuda(),
                attention_mask=concatenated_attention_mask.cuda(),
                images=concatenated_images.half().cuda()
            )

            pad_len = output.logits.shape[1]-concatenated_labels.shape[1]
            concatenated_labels = torch.nn.functional.pad(concatenated_labels, pad=(pad_len, 0), value=IGNORE_INDEX)
            win_size = win_input_ids.shape[0]
            rej_size = rej_input_ids.shape[0]

            per_token_logp, log_prob, average_log_prob = get_batch_logps(output.logits, concatenated_labels.cuda(), return_all=True)

            flag = per_token_logp.size(1) >= concatenated_input_ids.size(1) - 1

            if flag:
                win_logp, rej_logp = log_prob.split([win_size, rej_size])
                win_per_token_logp, rej_per_token_logp = per_token_logp.split([win_size, rej_size])
                win_average_log_prob, rej_average_log_prob = average_log_prob.split([win_size, rej_size])

                win_per_token_logp = win_per_token_logp.tolist()
                rej_per_token_logp = rej_per_token_logp.tolist()
                win_logp = win_logp.tolist()
                rej_logp = rej_logp.tolist()
                win_average_log_prob = win_average_log_prob.tolist()
                rej_average_log_prob = rej_average_log_prob.tolist()

                win_logp_list += win_logp
                win_avg_logp_list += win_average_log_prob
                win_per_token_logp_list += win_per_token_logp

                rej_logp_list += rej_logp
                rej_avg_logp_list += rej_average_log_prob
                rej_per_token_logp_list += rej_per_token_logp

            else:   # invalid logp
                win_logp_list += [None]
                win_avg_logp_list += [None]
                win_per_token_logp_list += [None]
                rej_logp_list += [None]
                rej_avg_logp_list += [None]
                rej_per_token_logp_list += [None]



    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list



def write_logp_to_preference_parquet(origin_data,
                                     logp_file,
                                     logps,
                                     # verification_info,
                                     args):
    out_data = []

    for index in range(len(origin_data)):
        line = origin_data[index]
        logp_data = logps[index]

        if logp_data[0] == [None]:
            continue

        new_line = copy.deepcopy(line)

        text = json.loads(new_line['text'])

        if 'logps' in text.keys():
            assert args.overwrite_logps, 'Found existing logp data, pass args.overwrite_logps=True to force overwritting'
            text['logps'] = logp_data
            new_line['text'] = json.dumps(text)
            # new_line['ver_info'] = json.dumps(ver)

        else:
            assert list(text.keys()) == ['question', 'chosen', 'rejected'], f'Undefined data structure, expecting [Q, Win, Rej], got {text.keys()}'
            text['logps'] = logp_data
            new_line['text'] = json.dumps(text)
            # new_line['ver_info'] = json.dumps(ver)

        out_data.append(new_line)

    df = pd.DataFrame(out_data)

    df.to_parquet(logp_file)

    return df


def inference_logp(model, tokenizer, hf_data,
                   logp_file,
                   image_processor, args):
    # model = model.to(dtype=torch.float16, device='cuda')
    dataset = PreferenceInferenceDataset(tokenizer=tokenizer,
                                         data=hf_data,
                                         image_processor=image_processor,
                                         args=args)
    collate_fn = partial(preference_collator_fn, tokenizer=tokenizer)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=1, shuffle=False)

    # print(len(dataloader), len(dataset))

    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list\
        = get_multimodal_sample_logps(model, dataloader)

    # world_size = torch.distributed.get_world_size()
    # merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    # for i in range(len(outputs)):
    #     torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
    #     merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]
    #
    # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list,\
    #     = merged_outputs
    #
    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list,
                     rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    df = write_logp_to_preference_parquet(dataset.data,
                                          logp_file,
                                          logps,
                                          # verification_info,
                                          args)

    del model
    return df
