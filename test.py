import numpy as np 
import torch
import torch.nn.functional as F
from pytorch_transformers import WEIGHTS_NAME, BertConfig,BertForNextSentencePrediction, BertTokenizer
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_examples_to_features(texts_a, tokenizer, texts_b=None,max_seq_length=128,
                                cls_token_at_end=False, pad_on_left=False,
                                cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    features = []
    for text_a,text_b in zip(texts_a,texts_b):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = tokenizer.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # logger.info("*** Example ***")
        # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def test(texts_a, texts_b, tokenizer, model):

    features = convert_examples_to_features(texts_a=texts_a,
                                            texts_b=texts_b,
                                            tokenizer=tokenizer, max_seq_length=32)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    test_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
                                        
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # for batch in tqdm(test_dataloader, desc="Test"):
    for batch in test_dataloader:
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            outputs = model(**inputs)
            # logger.info(outputs)
            # logger.info(torch.argmax(outputs[0],dim=-1))
            # 二分类，0代表是下一句 
            if torch.argmax(F.softmax(outputs[0],dim=-1),dim=-1).item()==0:
                logger.info([texts_a,texts_b])
                logger.info(F.softmax(outputs[0],dim=-1))
    return [texts_b, F.softmax(outputs[0],dim=-1)]


if __name__ == "__main__":
    import pickle
    import pandas as pd
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    model = BertForNextSentencePrediction.from_pretrained('out/')
    model.to('cuda')

    break_tag = pd.read_excel('/data/jh/notebooks/wanglei/1688/data/break_tag.xlsx',header=None)
    break_tag = list(break_tag[0])
    ret = []
    for tag in break_tag:
        text_a=['露出精致的锁骨和优美的天鹅颈']
        text_b=[tag]
        ret.append(test(text_a, text_b, tokenizer, model))
    

    


