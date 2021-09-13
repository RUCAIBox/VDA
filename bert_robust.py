# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : attack.py


import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES']= '4'
import random
import re

import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Dataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import argparse
import numpy as np
import timeout_decorator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)

def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word


def get_data_cls(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()[1:]
    features = []
    for i, line in enumerate(lines):
        split = line.strip('\n').split('\t')
        label = int(split[-1])
        seq = split[0]

        features.append([seq, label])
    return features


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def reader_data(filename, add=True):
    #return [[ori_text, adv_text, label], ... ,[]]
    f=open(filename, encoding='utf-8')
    output=[]
    for line in f:
        label, text = line.strip().split('\t')
        if add:
            label=int(label)-1
        else:
            label=int(label)
        output.append([text, label])
    if 'imdb' in filename or 'yelp' in filename:
        output=output[:300]
    return output

class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.ori_acc = None
        self.att_acc = 1
        self.changes = []

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words

class BERTDataset(Dataset):
    def __init__(self, inputs, masks, segs):
        self.inputs=inputs
        self.masks=masks
        self.segs=segs

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index], dtype=torch.long)
        attention_mask = torch.tensor(self.masks[index], dtype=torch.long)
        token_type_ids = torch.tensor(self.segs[index], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.inputs)

def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words)  # mask each words (not subwords!)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        token_type_ids = token_type_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)

    eval_data = BERTDataset(all_input_ids, all_masks, all_segs)
    # Run prediction for full data
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, seg_token, attn_mask  = (ele.cuda() for ele in batch)
        leave_1_prob_batch = tgt_model(masked_input, attn_mask, seg_token)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     + (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()
    return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

#@timeout_decorator.timeout(60)
def attack(feature, tgt_model, mlm_model, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={},
           use_bpe=1, threshold_pred_score=0.3):
    # MLM-process
    words, sub_words, keys = _tokenize(feature.seq, tokenizer)

    # original label
    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
    attention_mask = torch.tensor([1] * len(input_ids))
    seq_len = input_ids.size(0)
    orig_probs = tgt_model(input_ids.unsqueeze(0).to('cuda'),
                           attention_mask.unsqueeze(0).to('cuda'),
                           token_type_ids.unsqueeze(0).to('cuda')
                           )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.ori_acc = 0
        feature.att_acc = 0
        feature.success = -1
        return feature
    else:
        feature.ori_acc = 1

    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']

    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    # print(list_of_index)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))):
            feature.success = -2  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in filter_words:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue

        substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

        substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in filter_words:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.7:
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length,
                                           truncation=True)
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            token_type_ids = torch.tensor(inputs["token_type_ids"]).unsqueeze(0).to('cuda')
            attention_mask = torch.tensor([1] * len(input_ids)).unsqueeze(0).to('cuda')

            temp_prob = tgt_model(input_ids, attention_mask, token_type_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                feature.att_acc = 0
                return feature
            else:
                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
    feature.success = 2
    return feature

def save_to_original_BERT(model, save_file):
    print('model reload started!!')
    paras = torch.load(save_file)
    
    paras_new = {}
    for ele in paras:
        #if 'module.' in ele:	#'tgt_model.' in ele:
        if 'tgt_model.' in ele:
            paras_new[ele[10:]] = paras[ele]
        else:
            paras_new[ele] = paras[ele]
    
    model.load_state_dict(paras_new)
    print(model)

def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Robust/yelp.txt")
    parser.add_argument("--mlm_path", type=str, default="../bert_file", help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, default="saved/yelp_vda.pt", help="xxx classifier")

    parser.add_argument("--output_dir", type=str, default="data_defense", help="train file")
    parser.add_argument("--use_sim_mat", type=int, default=1,
                        help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--start", type=int, default=0, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, default=2000, help="end step, for multi-thread process")
    parser.add_argument("--num_label", type=int, default=2)
    parser.add_argument("--use_bpe", type=int, default=1)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--threshold_pred_score", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    data_path = str(args.data_path)
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_dir = str(args.output_dir)

    args = parser.parse_args()
    if 'ag' in data_path or 'yelp' in data_path:
        add = True
    else:
        add = False
    features = reader_data(data_path, add)

    # features = get_data_cls(data_path)

    num_label = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    threshold_pred_score = args.threshold_pred_score

    print('start process')

    # tokenizer_mlm = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)
    tokenizer_tgt = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)

    config_atk = BertConfig.from_pretrained(mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk)
    mlm_model.to('cuda')

    config_tgt = BertConfig.from_pretrained(mlm_path, num_labels=num_label)
    tgt_model = BertForSequenceClassification.from_pretrained(mlm_path, config=config_tgt)
    #tgt_model.load_state_dict(torch.load(tgt_path).state_dict())
    save_to_original_BERT(tgt_model, tgt_path)
    tgt_model.to('cuda')

    print('loading sim-embed')

    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('counter-fitted-vectors.txt', 'cos_sim_counter_fitting.npy')
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []
    out_f = open(os.path.join(output_dir, args.data_path.split('/')[-2] + '_adversaries.txt'), 'w')

    ori_acc=[]
    att_acc=[]
    q_num=[]
    perturb=[]

    with torch.no_grad():
        for index, feature in enumerate(features):
            # print(feature)
            seq_a, label = feature
            feat = Feature(seq_a, label)
            print('\r number {:d} '.format(index) + tgt_path, end='')
            # print(feat.seq[:100], feat.label)
            feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, k, batch_size=32, max_length=512,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=use_bpe, threshold_pred_score=threshold_pred_score)

                # print(feat.changes, feat.change, feat.query, feat.success)
            ori_acc.append(feat.ori_acc)
            att_acc.append(feat.att_acc)
            if feat.ori_acc==1:
                q_num.append(feat.query)
                perturb.append(feat.change/(len(feat.seq.strip().split())))

            # print(feat.changes, feat.change, feat.query, feat.success)
            new_line = str(feat.label) + '\t' + feat.seq.strip() + '\t' + feat.final_adverse.strip() + '\n'
            out_f.write(new_line)
            print('success', end='')
            features_output.append(feat)
        print('original accuracy is %f, attack accuracy is %f, query num is %f, perturb rate is %f'
              %(sum(ori_acc)/len(ori_acc), sum(att_acc)/len(att_acc), sum(q_num)/len(q_num), sum(perturb)/len(perturb)))



if __name__ == '__main__':
    run_attack()
