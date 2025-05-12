#!/usr/bin/env python
import os
import torch
import numpy as np
from typing import List

from glue_utils import convert_examples_to_seq_features, ABSAProcessor
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, WEIGHTS_NAME
from absa_layer import BertABSATagger
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from seq_utils import ot2bieos_ts, bio2ot_ts, tag2ts
from transformers.data.processors.utils import InputExample

# =================== CẤU HÌNH =====================
absa_home = "./bert-linear-laptop14-finetune"
ckpt = os.path.join(absa_home, "checkpoint-1500")
model_type = "bert"
model_name_or_path = "bert-base-uncased"
cache_dir = "./cache"
max_seq_length = 128
tagging_schema = "BIEOS"
# ================================================

# Init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# Khởi tạo model/tokenizer
config_class, model_class, tokenizer_class = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}[model_type]

print(f"Load model từ checkpoint: {ckpt}/{WEIGHTS_NAME}")
model = model_class.from_pretrained(ckpt)
tokenizer = tokenizer_class.from_pretrained(absa_home)
model.to(device)
model.eval()


def create_example_from_text(text: str, labels: List[str]):
    return InputExample(guid="user-input", text_a=text, text_b=None, label=labels)


def prepare_input(text: str):
    processor = ABSAProcessor()
    tokens = text.strip().split()
    dummy_labels = ['O'] * len(tokens)
    example = create_example_from_text(text, dummy_labels)
    examples = [example]
    label_list = processor.get_labels(tagging_schema)

    features = convert_examples_to_seq_features(
        examples=examples,
        label_list=label_list,
        tokenizer=tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=0,
        pad_on_left=False,
        pad_token_segment_id=0
    )

    words = [text.strip().split()]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    evaluate_label_ids = [f.evaluate_label_ids for f in features]

    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    return dataset, evaluate_label_ids, words


def predict(text: str):
    dataset, evaluate_label_ids, total_words = prepare_input(text)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1)

    if tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                            'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                            'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3,
                            'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    else:
        raise Exception(f"Invalid tagging schema {tagging_schema}")

    absa_id2tag = {v: k for k, v in absa_label_vocab.items()}

    for idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if model_type in ['bert', 'xlnet'] else None,
                'labels': batch[3]
            }
            outputs = model(**inputs)
            logits = outputs[1]

            if model.tagger_config.absa_type != 'crf':
                preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            else:
                mask = batch[1]
                preds = model.tagger.viterbi_tags(logits=logits, mask=mask)

            label_indices = evaluate_label_ids[idx]
            words = total_words[idx]
            pred_labels = preds[0][label_indices]
            assert len(words) == len(pred_labels)

            pred_tags = [absa_id2tag[label] for label in pred_labels]

            if tagging_schema == 'OT':
                pred_tags = ot2bieos_ts(pred_tags)
            elif tagging_schema == 'BIO':
                pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))

            p_ts_sequence = tag2ts(ts_tag_sequence=pred_tags)
            output_ts = []
            for t in p_ts_sequence:
                beg, end, sentiment = t
                aspect = words[beg:end + 1]
                output_ts.append('%s: %s' % (' '.join(aspect), sentiment))
            print("Input:", ' '.join(words))
            print("Output:", '\t'.join(output_ts))


if __name__ == "__main__":
    print("==== ABSA BERT Linear Inference ====")
    print("Nhập câu cần phân tích khía cạnh (hoặc gõ `exit` để thoát):")
    while True:
        text = input(">> ")
        if text.lower().strip() in ['exit', 'quit']:
            break
        if text.strip():
            predict(text)
