import os
import json
import argparse
import logging
from llava.utils import write_to_csv


def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        # if text.find('.') != -1:
        #     text = text.split('.')[0]

        text = text.replace(',', '')
        # text = text.replace('.', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    # print('TP\tFP\tTN\tFN\t')
    # print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    # logging.info('TP: {}\tFP: {}\tTN: {}\tFN: {}'.format(TP, FP, TN, FN))
    # logging.info('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))
    return TP, FP, TN, FN, f1, acc, precision, recall, yes_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--log-file", type=str, default='./playground/data/eval/pope/scores0.csv')
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    # logging.basicConfig(
    #     filename=args.log_file,
    #     filemode='a',
    #     format='%(asctime)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S',
    #     level=logging.INFO
    # )
    # logging.info("result file: {}".format(args.result_file))

    header = [
        'result file', 'category',
        'TP', 'FP', 'TN', 'FN',
        'f1', 'acc', 'precision', 'recall', 'yes_ratio']

    f1s = []
    accs = []

    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        # logging.info('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        TP, FP, TN, FN, f1, acc, precision, recall, yes_ratio = \
            eval_pope(cur_answers, os.path.join(args.annotation_dir, file))

        f1s.append(f1)
        accs.append(acc)
        write_to_csv(args.log_file, header,
                     [args.result_file.split('/')[-1], category,
                      TP, FP, TN, FN,
                      f1, acc, precision, recall, yes_ratio])
        print("====================================")
    write_to_csv(args.log_file, header,
                 [args.result_file.split('/')[-1], 'avg',
                  '', '', '', '',
                  sum(f1s)/len(f1s), sum(accs)/len(accs), '', '', ''])
