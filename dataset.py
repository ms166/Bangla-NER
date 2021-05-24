import numpy as np
import torch
from torch.utils import data
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

VOCAB = ('<PAD>', 'I-LOC', 'B-ORG', 'O', 'I-OBJ', 'I-PER', 'B-OBJ', 'I-ORG', 'B-LOC', 'B-PER')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, sents, tags_li):
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]

        # these chars make it difficult to figure out which token belongs to which word
        # chars to replace (hex): 0x200b 0x200c 0x200d 0x200e 0x200f 0xaf 0xb4
        #                       : \u200b \u200c \u200d \u200e \u200f \xaf \xb4
        for i, word in enumerate(words):
            if('\u200b' in word):
                words[i] = word.replace('\u200b', '-')
            if('\u200c' in word):
                words[i] = word.replace('\u200c', '-')
            if('\u200d' in word):
                words[i] = word.replace('\u200d', '-')
            if('\u200e' in word):
                words[i] = word.replace('\u200e', '-')
            if('\u200f' in word):
                words[i] = word.replace('\u200f', '-')
            if('\xaf' in word):
                words[i] = word.replace('\xaf', '-')
            if('\xb4' in word):
                words[i] = word.replace('\xb4', '-')

        tokenized = [words[0]] + tokenizer.tokenize(' '.join(words[1:-1])) + [words[-1]]
        is_heads = [0 for x in range(len(tokenized))]
        is_heads[0] = is_heads[-1] = 1
        for i, token in enumerate(tokenized):
            if(token[0] == '▁'):
                is_heads[i] = 1

        ids = tokenizer.convert_tokens_to_ids(tokenized)
        x = ids

        y = [tag2idx["<PAD>"] for x in range(len(tokenized))]
        index = 0
        try:
            for i in range(len(tokenized)):
                if(is_heads[i] == 1):
                    y[i] = tag2idx[tags[index]]
                    index += 1
            assert(index == len(tags))
        except:
            print("Parsing error in the following sentence:")
            print(words)
            print(tags)
            print(tokenized)
            print(is_heads)
            print(x)
            print(y)
            print()
            exit()

        seqlen = len(y)
        assert(len(words) == len(tags))
        assert(len(x) == len(y))
        assert(len(y) == len(is_heads))
        assert(len(y) == len(tokenized))
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    y = f(-2, maxlen)
    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens


