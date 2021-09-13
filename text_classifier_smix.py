import random
import os
os.environ['CUDA_VISIBLE_DEVICES']= '7'
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertLayer, BertForMaskedLM, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from transformers import BertForSequenceClassification, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
import argparse

import numpy as np
import torch.nn.functional as F

class EmbAdapterDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=args.max_length
        self.pad_token = 0
        self.mask_ids = 103

    def __getitem__(self, index):
        x, y=self.data[index]
        inputs = self.tokenizer.encode_plus(x, None, add_special_tokens=True, max_length=self.max_length, truncation=True)
        padding_length = self.max_length-len(inputs["input_ids"])
        input_ids = torch.tensor(inputs["input_ids"]+padding_length*[self.pad_token], dtype=torch.long)
        token_type_ids = torch.tensor(self.max_length*[0], dtype=torch.long)
        attention_mask = torch.tensor([1]*len(inputs["input_ids"])+padding_length*[0], dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, y

    def __len__(self):
        return len(self.data)

class BertEncoder4SentMix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4SentMix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None,
                attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    # hidden_states = l * hidden_states + (1-l)*hidden_states2
                    # attention_mask = attention_mask.long() | attention_mask2.long()
                    # sentMix: (bsz, len, hid)
                    hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1 - l) * hidden_states2[:, 0, :]

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs

class BertModel4SentMix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4SentMix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4SentMix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000,
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class SentMix(BertPreTrainedModel):
    def __init__(self, config):
        super(SentMix, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel4SentMix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.kl_criterion = nn.KLDivLoss()

        self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None, token_type_ids=None):
        if x2 is not None:
            outputs = self.bert(x, attention_mask, x2, attention_mask, l, mix_layer, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(x, attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]


        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        return logits, outputs

    def output_with_emb(self, embedding_output, extended_attention_mask, head_mask, encoder_extended_attention_mask):
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output) if self.bert.pooler is not None else None
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def adv_train(self, input_ids, attention_masks, token_type_ids, y, optim, scheduler, mlm_model, args):
        logits = self.forward(input_ids, attention_masks, token_type_ids=token_type_ids)[0]
        L_ori = nn.CrossEntropyLoss()(logits, y)

        batch_size = input_ids.size(0)
        idx = torch.randperm(batch_size)
        input_ids_2 = input_ids[idx]
        labels_2 = y[idx]
        attention_mask_2 = attention_masks[idx]
        ## convert the labels to one-hot
        labels = torch.zeros(batch_size, logits.size(-1)).cuda().scatter_(
            1, y.view(-1, 1), 1
        )
        labels_2 = torch.zeros(batch_size, logits.size(-1)).cuda().scatter_(
            1, labels_2.view(-1, 1), 1
        )
        l = np.random.beta(args.alpha, args.alpha)
        # l = max(l, 1-l) ## not needed when only using labeled examples
        mixed_labels = l * labels + (1 - l) * labels_2

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        logits = self.forward(input_ids, attention_masks, input_ids_2, attention_mask_2, l, mix_layer, token_type_ids=token_type_ids)[0]
        probs = torch.softmax(logits, dim=1)  # (bsz, num_labels)
        L_mix = F.kl_div(probs.log(), mixed_labels, None, None, 'batchmean')

        input_shape = input_ids.size()
        device = input_ids.device
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_masks, input_shape, device)

        encoder_extended_attention_mask = None
        head_mask = self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)

        probs = mlm_model(input_ids, attention_masks, token_type_ids)[0]
        probs = probs / torch.sum(probs, -1, keepdim=True)
        noise = torch.randn(probs.size(), device=device) * args.variance
        probs = torch.softmax(probs + noise.cuda(), -1)  # [B, Len, V]
        word_embs = self.bert.embeddings.word_embeddings.weight
        input_embeds = torch.matmul(probs, word_embs)
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=input_embeds,
        )
        vda_logits = self.output_with_emb(embedding_output, extended_attention_mask, head_mask,
                                      encoder_extended_attention_mask)
        vda_loss = 0.5 * self.kl_criterion(torch.log_softmax(logits, -1), torch.softmax(vda_logits, -1)) + \
                 0.5 * self.kl_criterion(torch.log_softmax(vda_logits, -1), torch.softmax(logits, -1))

        total_loss=L_ori+L_mix  #+vda_loss
        total_loss.backward()
        optim.step()
        #print({name:param.grad for name, param in self.tgt_model.classifier.named_parameters()})
        scheduler.step()
        self.zero_grad()

        return L_ori.cpu().item(), L_mix.cpu().item(), vda_loss.cpu().item()


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
    return output


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mr")
    parser.add_argument("--mlm_path", type=str, default="../bert_file", help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, default="../TextFooler/target_models/mr",
                        help="xxx classifier")
    parser.add_argument("--save_path", type=str, default="saved/mr_smix.pt", help="xxx mlm")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_label", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_warmup", type=float, default=0.05)
    parser.add_argument("--variance", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--mix-layers-set",
        nargs='+',
        default=[7, 9, 12],
        type=int,
        help="define mix layer set"
    )

    args = parser.parse_args()
    data_path = 'data/' + str(args.dataset)
    if args.dataset == 'ag' or args.dataset == 'yelp':
        add = True
    else:
        add = False
    train_features = reader_data(os.path.join(data_path, 'train.txt'), add)
    dev_features = reader_data(os.path.join(data_path, 'dev.txt'), add)
    test_features = reader_data(os.path.join(data_path, 'test.txt'), add)

    num_label = args.num_label

    print('start process')

    # tokenizer_mlm = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)
    mlm_model = BertForMaskedLM.from_pretrained(args.mlm_path).cuda()
    tokenizer_tgt = BertTokenizer.from_pretrained(args.mlm_path, do_lower_case=True)

    train_data = EmbAdapterDataset(args, train_features, tokenizer_tgt)
    dev_data = EmbAdapterDataset(args, dev_features, tokenizer_tgt)
    test_data = EmbAdapterDataset(args, test_features, tokenizer_tgt)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    print('start building model')
    config_atk = BertConfig.from_pretrained(args.mlm_path, num_labels=args.num_label)
    model = SentMix.from_pretrained(args.mlm_path, config=config_atk)
    model= model.cuda()

    params=model.parameters()
    #need_grad = lambda x: x.requires_grad
    optimizer = AdamW(
        params,
        lr=args.lr, eps=1e-8, weight_decay=0.01,
    )
    total_num = len(train_data) // args.batch_size * args.max_epoch
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup * total_num,
                                                                   num_training_steps=total_num)
    best_ratio=0
    for epoch in range(args.max_epoch):
        model.train()
        attack_ratio = 0
        total_num = 0
        total_loss = []
        for batch in tqdm(train_dataloader):
            model.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = [ele.cuda() for ele in batch]
            #origin_emb = model.produce_emb(input_ids, attention_mask, token_type_ids)
            loss, kl_loss, vda_loss = model.adv_train(input_ids, attention_mask, token_type_ids, labels,
                                                             optimizer,
                                                             scheduler, mlm_model, args)
            total_loss.append([loss, kl_loss, vda_loss])

        print('Epoch %d, the training ce loss is %f, mix loss is %f, vda loss is %f, the total number is %f'
              % (epoch, sum([ele[0] for ele in total_loss]) / len(total_loss),
                 sum([ele[1] for ele in total_loss]) / len(total_loss),
                 sum([ele[2] for ele in total_loss]) / len(total_loss), total_num))

        attack_ratio = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids, token_type_ids, attention_mask, y = [ele.cuda() for ele in batch]
                probs = model(input_ids, attention_mask, token_type_ids)[0]  # , token_type_ids)[0]
                argmax_probs = torch.argmax(probs, dim=-1)
                success_num = torch.sum((argmax_probs == y).float()).cpu().item()
                attack_ratio += success_num
                total_num += input_ids.size(0)
        if best_ratio<attack_ratio:
            torch.save(model.state_dict(), args.save_path)
            print('--------save once-----------')
            best_ratio=attack_ratio
        print('The dev set defense success attack ratio is %f, the total number is %f' % (attack_ratio / total_num, total_num))

        attack_ratio = 0
        total_num = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, token_type_ids, attention_mask, y = [ele.cuda() for ele in batch]
                probs = model(input_ids, attention_mask, token_type_ids)[0]  # , token_type_ids)[0]
                argmax_probs = torch.argmax(probs, dim=-1)
                success_num = torch.sum((argmax_probs == y).float()).cpu().item()
                attack_ratio += success_num
                total_num += input_ids.size(0)
        print('The test set defense success attack ratio is %f, the total number is %f' % (attack_ratio / total_num, total_num))

if __name__ == '__main__':
    run()
