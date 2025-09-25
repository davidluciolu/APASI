import os
import json
from PIL import Image
import pandas as pd
# from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
import random
random.seed(42)
import re

import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import argparse

import torch
from transformers import AutoTokenizer, AutoConfig
# from muffin import Beit3LlavaLlamaForCausalLM
# from muffin.conversation import conv_templates
# from muffin.utils import disable_torch_init
from transformers import StoppingCriteria


# from CHAIR https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py
class ClosedSetNegConstructor:
    def __init__(self, sym_file, cap_file,
                 object_vocab_mode='chair',
                 processed_caption_file='data/processed_captions.json',
                 re_process=False,
                 **kwargs):

        self.sym_file = sym_file
        self.cap_file = cap_file
        self.processed_caption_file = processed_caption_file
        self.lemmatizer = WordNetLemmatizer()
        self.object_vocab_mode = object_vocab_mode
        if self.object_vocab_mode.lower() == 'chair':
            self._get_object_vocab_chair()
            self.find_object_words = self.find_object_words_chair
        elif self.object_vocab_mode.lower() == 'lvis':
            self._get_object_vocab_lvis()
            self.find_object_words = self.find_object_words_lvis
        else:
            raise NotImplementedError(f'object vocab mode {self.object_vocab_mode} not implemented')

        if os.path.exists(self.processed_caption_file) and (not re_process):
            data = json.load(open(self.processed_caption_file))
            self.processed_captions = data['processed_captions']
            self.object_freq = data['node_object_freq']
            self.object_co_freq = data['node_object_co_freq']
        else:
            self._process_captions()

    def replace_objects(self, replace_rate=0.3, object_threshold=2, replace_strategy='mixed'):
        assert replace_strategy in ['mixed', 'random', 'popular', 'adversarial']

        captions = json.load(open(self.cap_file))

        columns = ['ds_name', 'text', 'origin_dataset', 'origin_split', 'idx', 'image_path']

        ds_name = self.cap_file.split('/')[-1].replace('.json', '')
        origin_dataset = self.cap_file.split('/')[-1].replace('.json', '')
        origin_split = json.dumps({"model": "GPT-4 anno", "type": "detailed_description"})
        coco_dir = '../../pub_dataset/coco/train2017'

        df = pd.DataFrame(columns=columns)

        for cap_id, item in tqdm(self.processed_captions.items()):
            caption = item['caption']
            objects = item['node_objects']
            indexes = item['indexes']
            source_index = item['source_index']
            num_objects = item['num_objects']
            replace_num = round(replace_rate * num_objects)

            if replace_num < object_threshold or replace_num >= num_objects:
                continue
            else:
                non_exist_objects = {k: v for k, v in self.object_freq.items() if k not in objects}
                # non_exist_objects = sorted(non_exist_objects.items(), key=lambda x: x[1], reverse=True)

                if replace_strategy == 'mixed':
                    strats = random.choices(['random', 'popular', 'adversarial'], k=replace_num)
                else:
                    strats = [replace_strategy for i in range(replace_num)]

                replaced_ids = random.sample(range(num_objects), replace_num)
                unchange_ids = list(set(range(num_objects)) - set(replaced_ids))
                unchange_objects = list(set([objects[j] for j in unchange_ids]))

                words = word_tokenize(caption)

                offset = 0
                for i, strat in zip(replaced_ids, strats):
                    # decide replaced words, in list
                    replaced_object, flag = self._decide_replaced_object(strat, unchange_objects, non_exist_objects)

                    replaced_word = random.choice(self.synonym_dict[replaced_object]).split(' ')
                    start, end = indexes[i][0], indexes[i][-1]
                    offset += len(replaced_word) - (end - start + 1)
                    words[start:end + 1] = replaced_word

                question = captions[source_index]['conversations'][0]['value']
                image_path = os.path.join(coco_dir, captions[source_index]['image'])

                text = json.dumps({'question': question, 'chosen': caption, 'rejected': ' '.join(words)})
                entry = {
                    'ds_name': ds_name,
                    # 'image': image,
                    'text': text,
                    'origin_dataset': origin_dataset,
                    'origin_split': origin_split,
                    'idx': cap_id,
                    'image_path': image_path
                }
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)


        print(f'orginal sample: {len(self.processed_captions)}, collected: {len(df)}')

        # if len(fake_adversarial) != 0:
        #     print('fake adversarial')
        # print(fake_adversarial)

        folder_name = os.path.join('../LLaVA/playground/data/neg_data',
                                   f'{origin_dataset}_{self.object_vocab_mode}_{replace_strategy}_replace_{replace_rate}_{object_threshold}')
        os.makedirs(folder_name, exist_ok=True)
        file_name = os.path.join(folder_name, 'replaced_temp_0.parquet')
        print('saved to: ', os.path.abspath(file_name))
        df.to_parquet(file_name)


    def _get_processed_captions(self):
        return self.processed_captions

    def _get_object_stats(self):
        return self.object_freq, self.object_co_freq

    def _process_captions(self):
        captions = json.load(open(self.cap_file))

        self.processed_captions = {}
        object_freq = {}
        object_co_freq = {}
        for s_i, item, in enumerate(tqdm(captions)):
            # if 'detail_23k' in self.cap_file:
            #     caption = item['conversations'][1]['value']
            # else:
            #     caption = item
            caption = item['conversations'][1]['value']
            objects, indexes = self.find_object_words(caption)
            assert len(objects) == len(indexes)
            self.processed_captions.update({
                item['id']:
                {
                'source_index': s_i,
                'caption': caption,
                'node_objects': objects,
                'indexes': indexes,
                'num_objects': len(objects)}
            })

            for o in set(objects):
                object_freq[o] = object_freq.get(o, 0) + 1
                o_co_freq = object_co_freq.get(o, {})
                for oo in set(objects):
                    if oo != o:
                        o_co_freq[oo] = o_co_freq.get(oo, 0) + 1
                object_co_freq[o] = o_co_freq

        self.object_freq = dict(sorted(object_freq.items(), key=lambda x: x[1], reverse=True))
        self.object_co_freq = {k: dict(sorted(v.items(), key=lambda x: x[1], reverse=True))for k, v in object_co_freq.items()}

        avg_num_obj = sum([cap['num_objects'] for cap in self.processed_captions.values()])/len(self.processed_captions)

        json.dump(
            {
                'processed_captions':self.processed_captions,
                'node_object_freq':self.object_freq,
                'node_object_co_freq':self.object_co_freq,
                'synonym_dict':self.synonym_dict,
                'avg_object_per_cap':avg_num_obj,       # 8.774612736660929 for chair
        },
            open(self.processed_caption_file, 'w'),
            indent=4
        )

    def _decide_replaced_object(self, strat, unchange_objects, non_exist_objects):
        fake_ad_flag = False

        if strat == "random":
            replaced_object = random.choice(list(non_exist_objects.keys()))
        elif strat == "popular":
            replaced_object = list(non_exist_objects.keys())[0]
        elif strat == "adversarial":
            random.shuffle(unchange_objects)
            j = 0
            replaced_object = ''
            # traverse unchange objects, find co-occur with unchanged but non-exist object
            while j < len(unchange_objects):
                co_freq = self.object_co_freq.get(unchange_objects[j], {})
                if len(co_freq) == 0:  # unchanged have no co-occur, can't find for this unchanged
                    j += 1
                else:
                    for co_o in co_freq.keys():
                        if co_o in non_exist_objects.keys():  # unchanged but non-exist found
                            replaced_object = co_o
                            break
                    if replaced_object == '':  # co-occur all exist, can't find for this unchanged
                        j += 1
                    else:
                        break
            if replaced_object == '':  # can't find for all unchanged, take a random instead
                replaced_object = list(non_exist_objects.keys())[0]
                fake_ad_flag = True
        else:
            raise ValueError(f'unsupported replace strategy{strat}')

        return replaced_object, fake_ad_flag


    def _get_object_vocab_chair(self):
        # read in synonyms
        synonyms = open(self.sym_file).readlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.all_objects = []  # mscoco objects and *all* synonyms
        self.synonym_dict = {}
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            synonym = [s.strip() for s in synonym]
            self.synonym_dict[synonym[0]] = synonym
            self.all_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                             'traffic signal',
                             'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball',
                             'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone',
                             'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer',
                             'hot dog', 'teddy bear', 'train track']

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal',
                        'cub']
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        # double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        # check if all double in sym list
        # for k, v in self.double_word_dict.items():
        #     if v not in self.all_objects:
        #         print(v)
        # pass

    def _get_object_vocab_lvis(self):
        def process_lvis_word(s):
            s = s.replace('_', ' ')
            s = re.sub(r'\s*\([^)]*\)', '', s)
            s = s.strip()
            return s

        synonyms = json.load(open(self.sym_file))
        discard_list = ['can', 'spread']
        self.all_objects = []
        self.synonym_dict = {}
        self.inverse_synonym_dict = {}
        self.double_word_dict = {}
        for item in synonyms:
            node_word = item['name']
            words = []
            for word in item['synonyms']:
                word = process_lvis_word(word)
                if word in discard_list:
                    continue
                self.inverse_synonym_dict[word] = node_word
                word_split = word.split(' ')
                if len(word_split) == 2:
                    self.double_word_dict[word] = word
                words.append(word)

            self.synonym_dict[node_word] = words
            self.all_objects.extend(words)

    def find_object_words_lvis(self, caption):
        return self.find_object_words_chair(caption)

    def find_object_words_chair(self, caption):

        '''
        Input: caption
        Output: indexes of objects in captions, corresponding node words
        '''

        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('N'):
                return wordnet.NOUN
            return None

        # standard preprocessing
        # print(caption)
        words = word_tokenize(caption)

        tagged_words = pos_tag(words)
        lemmatized_words = []
        for word, tag in tagged_words:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag == wordnet.NOUN:
                lemmatized_word = self.lemmatizer.lemmatize(word, wn_tag)
            else:
                lemmatized_word = word
            lemmatized_words.append(lemmatized_word)

        indexes = []
        node_words = []
        i = 0
        while i < len(lemmatized_words):
            double_word = ' '.join(lemmatized_words[i:i + 2])
            # is double word
            if double_word in self.double_word_dict:
                node_words.append(self.inverse_synonym_dict[self.double_word_dict[double_word]])
                indexes.append((i, i + 1))
                i += 2
            else:
                # is single word
                if lemmatized_words[i] in self.all_objects:
                    node_words.append(self.inverse_synonym_dict[lemmatized_words[i]])
                    indexes.append((i, i))
                i += 1

        return node_words, indexes

if __name__ == '__main__':
    # nc = ClosedSetNegConstructor(
    #     sym_file='data/LVIS_CATEGORIES.json',
    #     cap_file='data/detail_23k.json',
    #     processed_caption_file='data/detail_23k_lvis_processed.json',
    #     object_vocab_mode='lvis',
    #     re_process=True
    # )

    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, default="detail_23k_llava_ia0_gen")
    parser.add_argument("--obj_vocab_mode", type=str, default='lvis')
    parser.add_argument("--replace_obj", type=bool, default=False)
    parser.add_argument("--replace_rate", type=float, default=0.5)
    parser.add_argument("--object_threshold", type=int, default=4)
    args = parser.parse_args()

    data_folder = './playground/data/neg_data/'
    cap_file = os.path.join(data_folder, args.cap_file + '.json')
    processed_caption_file = os.path.join(data_folder, f'{args.cap_file}_{args.obj_vocab_mode.lower()}_processed.json')
    if args.obj_vocab_mode.lower() == 'lvis':
        sym_file = os.path.join(data_folder, 'LVIS_CATEGORIES.json')
    elif args.obj_vocab_mode.lower() == 'chair':
        sym_file = os.path.join(data_folder, 'synonyms.txt')
    else:
        raise ValueError(f'not supported mode: {args.obj_vocab_mode}')


    nc1 = ClosedSetNegConstructor(
        sym_file=sym_file,
        cap_file=cap_file,
        processed_caption_file=processed_caption_file,
        object_vocab_mode=args.obj_vocab_mode,
        re_process=False if os.path.exists(processed_caption_file) else True
    )

    if args.replace_obj:
        # nc.replace_objects(replace_strategy='mixed', replace_rate=0.5, object_threshold=4)
        # nc.replace_objects(replace_strategy='random', replace_rate=0.5, object_threshold=4)
        # nc.replace_objects(replace_strategy='popular', replace_rate=0.5, object_threshold=4)
        # nc.replace_objects(replace_strategy='adversarial', replace_rate=0.5, object_threshold=4)
        nc1.replace_objects(replace_strategy='adversarial',
                            replace_rate=args.replace_rate,
                            object_threshold=args.object_threshold)
