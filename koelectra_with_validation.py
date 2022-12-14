# -*- coding: utf-8 -*-
"""Koelectra_with_validation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sH_k04G8eOQqFdzCXPduU4_DqJDmppli

# Colab Mount & Import Libraries
"""

from google.colab import drive
drive.mount('/content/drive')

# train_file = "train.json"
# test_file = "test.json"
# result_path = "/content/drive/MyDrive/Colab Notebooks/KDT Project 2"
# blank_file = "blank.csv"

train_file = "data/train.json"
add_train_file = 'data/017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_span_extraction.json'
add_train_rev_file = "data/data_extraction_answer_len20.json"
test_file = "data/test.json"
result_path = "sub/"
blank_file = "data/blank.csv"

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/GoormProject/GoormProject2/

!pip install transformers
!pip install sentencepiece
!pip install wandb



import os
import random
import math
import csv
import json
from tqdm.notebook import tqdm
from easydict import EasyDict as edict
from statistics import mean
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import wandb

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    AutoConfig,
    AdamW,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    ElectraModel,
    ElectraTokenizer,
    ElectraForQuestionAnswering, 
    AutoModelForQuestionAnswering, 
    AutoTokenizer
)



"""# Config"""

args = edict({'w_project': 'KoElectra',
              'w_entity': 'reodreamer', # WandB ID
              'learning_rate': 5e-5,
              'batch_size': {'train': 128,
                             'eval': 4,
                             'test': 128},
              'accumulate': 32,
              'epochs': 10,
              'seed': 42,
              'model_name': 'monologg/koelectra-base-v3-discriminator',
              'max_length': 512,
              'EarlyStopping' : True,
              'patience' : 5,
              })
args['NAME'] = ''f'koelectra_ep{args.epochs}_lr{args.learning_rate}_{random.randrange(0, 1024)}'
print(args.NAME)

"""# Wandb"""

do_wandb = True

if do_wandb: wandb.login()

if do_wandb: wandb.init(project = args.w_project, entity = args.w_entity)

if do_wandb:
  wandb.run.name = args.NAME
  wandb.config.learning_rate = args.learning_rate
  wandb.config.epochs = args.epochs
  wandb.config.batch_size = args.batch_size



"""# Setting Model"""

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
# model.to(device)

model.cuda();

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

"""# Preprocessing

## KoMRC
"""

from typing import List, Tuple, Dict, Any
import json
import random
import copy

class KoMRC:
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))
        
        return cls(data, indices)

##################################################################
    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, train_raio: float=.8,test_ratio: float=.5, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[:int(len(indices) * train_raio)]
        split_indices = indices[int(len(indices) * train_raio):]
        eval_indices = split_indices[:int(len(split_indices) * test_ratio)]
        test_indices = split_indices[int(len(split_indices) * test_ratio):]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices), cls(dataset._data, test_indices)
################################################################################

    @classmethod
    def extraction(cls, dataset, data_number: int, save=False):
        '''
        첫 번째 인자 : KoMRC의 인스턴스
        두 번째 인자 : 추출할 data 개수 (단, data단위로 추출, indices단위가 아님)
        반환값 : KoMRC의 인스턴스
        '''
        if data_number > len(dataset):
            raise Exception("입력한 인자가 데이터셋 크기보다 큽니다.")
        else:
            data = copy.deepcopy(dataset._data)
            data['data'] = random.sample(data['data'], data_number)
            indices = []
            for d_id, document in enumerate(data['data']):
                for p_id, paragraph in enumerate(document['paragraphs']):
                    for q_id, _ in enumerate(paragraph['qas']):
                        indices.append((d_id, p_id, q_id))
            
            # 잘라낸 데이터를 저장
            if save:
                with open("cut_result.json", "w") as write_file:
                    json.dump(data, write_file, indent=4)
        
        return cls(data, indices)

    def __add__(self, other) :
        data = copy.deepcopy(self._data)
        data['data'] = self._data['data'] + other._data['data']
        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))
        return KoMRC(data, indices)
    

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        context = paragraph['context']
        qa = paragraph['qas'][q_id]

        guid = qa['guid']
        question = qa['question']
        answers = qa['answers']


        return {
            'guid': guid,
            'context': context,
            'question': question,
            'answers': answers
        }

    def __len__(self) -> int:
        return len(self._indices)

dataset1 = KoMRC.load(train_file) # train json 불러오고
dataset2 = KoMRC.load(add_train_rev_file) # 사전에 형식을 train.json 파일과 똑같이 만들어 놓은 TL_span_extraction2.json 파일을 불러오고
dataset2 = KoMRC.extraction(dataset2, int(len(dataset1)*.3))
dataset = dataset1 + dataset2

print(len(dataset1))
print(len(dataset2))
print(len(dataset))

from pprint import pprint
dataset = KoMRC.load(train_file)
pprint(dataset[0])

"""## TokenizedKoMRC"""

from typing import Generator
class TokenizedKoMRC(KoMRC):
    def __init__(self, data, indices: List[Tuple[int, int, int]]) -> None:
        super().__init__(data, indices)
        self._tokenizer  = tokenizer

    
    def _tokenize_with_position(self, sentence: str) -> List[Tuple[str, Tuple[int, int]]]:
        position = 0
        tokens = []
        sentence_tokens = []

        for word in sentence.split(): 
            if '[UNK]' in tokenizer.tokenize(word):
                sentence_tokens.append(word)
            else:
                sentence_tokens += tokenizer.tokenize(word)
        
        for morph in sentence_tokens:
            if len(morph) > 2:
                if morph[:2] == '##':
                    morph = morph[2:]

            position = sentence.find(morph, position)
            tokens.append((morph, (position, position + len(morph)))) # morps의 시작 위치, 형태소의 길이 만큼을 더하면 그것이 캐릭터의 끝 위치.
            position += len(morph) # 포지션 업데이트 
        
        return tokens
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        # sample = {'guid': guid, 'context': context, 'question': question, 'answers': answers}

        context, position = zip(*self._tokenize_with_position(sample['context']))
        context, position = list(context), list(position)

        question = self._tokenizer.tokenize(sample['question'])

        if sample['answers'] is not None:
            answers = []
            # 추가된 코드
            local_shortest_answer_len = 200; local_shortest_answer_idx = 0
            for i, answer in enumerate(sample['answers']):
                if len(answer['text'])<local_shortest_answer_len:
                  local_shortest_answer_len = len(answer['text'])
                  local_shortest_answer_idx=i
            answer=sample['answers'][local_shortest_answer_idx]
            for start, (position_start, position_end) in enumerate(position):
                if position_start <= answer['answer_start'] < position_end:
                    break
            # else:
            #     print(context, answer)
            #     # print(answer['guid'])
            #     # print(answer['answer_start'])
            #     raise ValueError("No mathced start position")

            target = ''.join(answer['text'].split(' '))
            source = ''
            for end, morph in enumerate(context[start:], start):
                source += morph
                if target in source:
                    break
            # else:
            #     print(context, answer)
            #     # print(answers['guid'])
            #     # print(answers['answer_start'])
            #     raise ValueError("No Matched end position")

            answers.append({'start': start, 'end': end})
            answer_text = sample['answers'][0]['text']

        else:
            answers = None
            answer_text = None
        
        return {
            'guid': sample['guid'],
            'context_original': sample['context'],
            'context_position': position,
            'question_original': sample['question'],
            'context': context,
            'question': question,
            'answers': answers,
            'answers_text': answer_text
        }

dataset = TokenizedKoMRC.load(train_file)
train_dataset, dev_dataset, stest_dataset = TokenizedKoMRC.split(dataset)
print("Number of Samples:", len(dataset))
print("Number of Train Samples:", len(train_dataset))
print("Number of Dev Samples:", len(dev_dataset))
print(dev_dataset[0])

"""## Indexer"""

class Indexer:
    def __init__(self, vocabs: List[str], max_length: int=args.max_length):
        self.max_length = args.max_length
        self.vocabs = vocabs

    @property
    def vocab_size(self):
        return len(self.vocabs)
    @property
    def pad_id(self):
        return tokenizer.vocab['[PAD]']
    @property
    def unk_id(self):
        return tokenizer.vocab['[UNK]']
    @property
    def cls_id(self):
        return tokenizer.vocab['[CLS]']
    @property
    def sep_id(self):
        return tokenizer.vocab['[SEP]']


    def sample2ids(self, sample: Dict[str, Any],) -> Dict[str, Any]:
        context = [tokenizer.convert_tokens_to_ids(token) for token in sample['context']]
        question = [tokenizer.convert_tokens_to_ids(token) for token in sample['question']]

        context = context[:self.max_length-len(question)-3]             # Truncate context
        
        input_ids = [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)

        if sample['answers'] is not None:
            answer = sample['answers'][0]
            start = min(len(question) + 2 + answer['start'], self.max_length - 1)
            end = min(len(question) + 2 + answer['end'], self.max_length - 1)
        else:
            start = None
            end = None

        return {
            'guid': sample['guid'],
            'context': sample['context_original'],
            'question': sample['question_original'],
            'position': sample['context_position'],
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'start': start,
            'end': end,
            'answers_text': sample['answers_text']
        }

indexer = Indexer(list(tokenizer.vocab.keys()))
print(indexer.sample2ids(dev_dataset[0]))

"""## IndexerWrappedDataset"""

class IndexerWrappedDataset:
    def __init__(self, dataset: TokenizedKoMRC, indexer: Indexer) -> None:
        self._dataset = dataset
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._indexer.sample2ids(self._dataset[index])
        sample['attention_mask'] = [1] * len(sample['input_ids'])

        return sample

indexed_train_dataset = IndexerWrappedDataset(train_dataset, indexer)
indexed_dev_dataset = IndexerWrappedDataset(dev_dataset, indexer)
indexed_stest_dataset = IndexerWrappedDataset(stest_dataset, indexer)

sample = indexed_dev_dataset[0]
# print(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'], sample['start'], sample['end'])
for i in sample:
  print(i,sample[i])

with open("data/indexed_train_dataset_koel.json", "w", encoding='utf-8') as write_file:
    json.dump(list(indexed_train_dataset), write_file, indent=4)

with open("data/indexed_dev_dataset_koel.json", "w", encoding='utf-8') as write_file:
    json.dump(list(indexed_dev_dataset), write_file, indent=4)

with open("data/indexed_stest_dataset_koel.json", "w", encoding='utf-8') as write_file:
    json.dump(list(indexed_stest_dataset), write_file, indent=4)

# with torch.no_grad(), open('data/indexed_train_dataset_koel.csv', 'w') as fd:
#     writer = csv.writer(fd)
#     writer.writerow(['guid', 'context', 'question', 'position','input_ids', 'token_type_ids','start', 'end', 'answers_text', 'attention_mask'])

# with torch.no_grad(), open('data/indexed_dev_dataset_koel.csv', 'w') as fd:
#     writer = csv.writer(fd)
#     writer.writerow(['guid', 'context', 'question', 'position','input_ids', 'token_type_ids','start', 'end', 'answers_text', 'attention_mask'])

"""## Collator"""

import torch
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = { 
            key: [sample[key] for sample in samples] # 키 단위로 묶기
            for key in samples[0]
        }

        for key in 'start', 'end': # start랑 end는 숫자로 들어가서 패딩 필요 없음
            if samples[key][0] is None:
                samples[key] = None
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)
        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            samples[key] = pad_sequence( # 패딩
                [torch.tensor(sample, dtype=torch.long) for sample in samples[key]],
                batch_first=True, padding_value=self._indexer.pad_id
            )
        return samples

collator = Collator(indexer)
train_loader = DataLoader(indexed_train_dataset,
                          batch_size = args.batch_size.train // args.accumulate,
                          shuffle = True,
                          collate_fn = collator,
                          num_workers = 2)

dev_loader = DataLoader(indexed_dev_dataset,
                        batch_size = args.batch_size.eval,
                        shuffle = False,
                        collate_fn = collator,
                        num_workers = 2)

batch = next(iter(dev_loader))
print(batch['input_ids'].shape)
print(batch['input_ids'])
print(list(batch.keys()))
# [CLS]는 2

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
gc.collect()
torch.cuda.empty_cache()



os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""## Edit Distance"""

# 출처: https://lsh424.tistory.com/78
def edit_distance(s:str, t: str):
    m = len(s)+1
    n = len(t)+1
    D = [[0]*m for _ in range(n)]
    D[0][0] = 0
    
    for i in range(1,m):
        D[0][i] = D[0][i-1] + 1
    
    for j in range(1,n):
        D[j][0] = D[j-1][0] + 1
    
    for i in range(1,n):
        for j in range(1,m):
            cost = 0

            if s[j-1] != t[i-1]:
                cost = 1
            
            D[i][j] = min(D[i][j-1] + 1,D[i-1][j] + 1, D[i-1][j-1] + cost)
    
    return D[n-1][m-1]

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=args.patience, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

"""# Train"""

train_losses = []
dev_losses = []

train_loss = []
dev_loss = []

loss_accumulate = 0.

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

best_model = [-1, int(1e9)]

for epoch in range(args.epochs):
    print("Epoch", epoch, '===============================================================================================================')

    # Train    
    progress_bar_train = tqdm(train_loader, desc='Train')
    for i, batch in enumerate(progress_bar_train, 1):
        del batch['guid'], batch['context'], batch['question'], batch['position'], batch['answers_text']
        batch = {key: value.cuda() for key, value in batch.items()}
        
        start = batch.pop('start')
        end = batch.pop('end')
        
        output = model(**batch)

        start_logits = output.start_logits
        end_logits = output.end_logits
        
        loss = (F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)) / args.accumulate
        loss.backward()

        loss_accumulate += loss.item()

        del batch, start, end, start_logits, end_logits, loss
        
        if i % args.accumulate == 0:
            # clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)

            train_loss.append(loss_accumulate)
            progress_bar_train.set_description(f"Train - Loss: {loss_accumulate:.3f}")
            loss_accumulate = 0.
        else:
            continue

        if i % int(len(train_loader) / (args.accumulate * 25)) == 0:
            # Evaluation
            for batch in dev_loader:
                del batch['guid'], batch['context'], batch['question'], batch['position'], batch['answers_text']
                batch = {key: value.cuda() for key, value in batch.items()}

                start = batch.pop('start')
                end = batch.pop('end')
                
                model.eval()
                with torch.no_grad():
                    output = model(**batch)
                
                    start_logits = output.start_logits
                    end_logits = output.end_logits
                model.train()

                loss = F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)

                dev_loss.append(loss.item())

                del batch, start, end, start_logits, end_logits, loss

            train_losses.append(mean(train_loss))
            dev_losses.append(mean(dev_loss))
            train_loss = []
            dev_loss = []

            
            if dev_losses[-1] <= best_model[1]:
                best_model = (epoch, dev_losses[-1])
                model.save_pretrained(f'models/{args.NAME}_{epoch}')
                # print(f'model saved!!\nvalid_loss: {dev_losses[-1]}')
                
            if do_wandb: wandb.log({"train_loss": train_losses[-1], "valid_loss": dev_losses[-1]})
            
            early_stopping(dev_losses[-1], model)

            if early_stopping.early_stop :
                print("Early stopping")
                break 

    print(f"Train Loss: {train_losses[-1]:.3f}")
    print(f"Valid Loss: {dev_losses[-1]:.3f}")
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

len(dev_losses)

import matplotlib.pyplot as plt

t = list(range(1, len(dev_losses)+1))
plt.plot(t, train_losses, label="Train Loss")
plt.plot(t, dev_losses, label="Dev Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""# Test"""

test_dataset = TokenizedKoMRC.load(test_file)
indexer_test = Indexer(list(tokenizer.vocab.keys()))
indexed_test_dataset = IndexerWrappedDataset(test_dataset, indexer_test)
print("Number of Test Samples", len(test_dataset))
# print(test_dataset[0])

print(best_model[0])
print(f'models/{args.NAME}_{best_model[0]}')

# 저장된 베스트 모델 가져오기 
model = AutoModelForQuestionAnswering.from_pretrained(f'models/{args.NAME}_{best_model[0]}')
model.cuda();

for i in indexed_train_dataset[1]:
  print(i,':',indexed_train_dataset[1][i])

for idx, sample in zip(range(1, 4), indexed_train_dataset):
    print(f'------{idx}------')
    print('Context:', sample['context'])
    print('Question:', sample['question'])
    
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="cuda")
        for key in ("input_ids", "token_type_ids")
    ]
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

    start_logits = output.start_logits
    end_logits = output.end_logits
    start_logits.squeeze_(0), end_logits.squeeze_(0)
    
    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

    index = torch.argmax(probability).item()
    
    start = index // len(end_prob)
    end = index % len(end_prob)
    
    start_str = sample['position'][start][0]
    end_str = sample['position'][end][1]
    print(start_str,end_str)
    pred_ans = sample['context'][start_str:end_str]
    print('Predicted Answer:', pred_ans)
    print('Real Answer:',sample['answers_text'])
    print('Edit Distance:',edit_distance(pred_ans,sample['answers_text']))

for idx, sample in zip(range(1, 4), indexed_dev_dataset):
    print(f'------{idx}------')
    print('Context:', sample['context'])
    print('Question:', sample['question'])
    
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="cuda")
        for key in ("input_ids", "token_type_ids")
    ]
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

    start_logits = output.start_logits
    end_logits = output.end_logits
    start_logits.squeeze_(0), end_logits.squeeze_(0)
    
    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

    index = torch.argmax(probability).item()
    
    start = index // len(end_prob)
    end = index % len(end_prob)
    
    start_str = sample['position'][start][0]
    end_str = sample['position'][end][1]
    print(start_str,end_str)
    pred_ans = sample['context'][start_str:end_str]
    print('Predicted Answer:', pred_ans)
    print('Real Answer:',sample['answers_text'])
    print('Edit Distance:',edit_distance(pred_ans,sample['answers_text']))

pred_answers = []
real_answers = []

edit_dists = []
for sample in tqdm(indexed_dev_dataset):
    input_ids, token_type_ids = [
        torch.tensor(sample[key], dtype=torch.long, device="cuda")
        for key in ("input_ids", "token_type_ids")
    ]
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

    start_logits = output.start_logits
    end_logits = output.end_logits
    start_logits.squeeze_(0), end_logits.squeeze_(0)
    
    start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
    end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

    probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

    index = torch.argmax(probability).item()
    
    start = index // len(end_prob)
    end = index % len(end_prob)
    
    start_str = sample['position'][start][0]
    end_str = sample['position'][end][1]
    pred_ans = sample['context'][start_str:end_str]
    if len(pred_ans)>20: pred_ans=''
    edit_dists.append(edit_distance(pred_ans,sample['answers_text']))
    pred_answers.append(pred_ans); real_answers.append(sample['answers_text'])

max(len(i) for i in pred_answers)

# 실제 답변 길이에 대해 예측 길이
pred_arr = [[0]*20 for _ in range(21)]
for i in range(len(pred_answers)):
  if len(real_answers[i])>20: continue
  pred_arr[len(real_answers[i])][len(pred_answers[i])]+=1
plt.matshow(pred_arr)
plt.xlabel('Length of Real Answer')
plt.ylabel('Length of Predicted Answer')
plt.colorbar()
plt.show()

import math
squared_pred_arr=[[0]*20 for _ in range(21)]
for i in range(21):
  for j in range(20):
    squared_pred_arr[i][j] = math.log(pred_arr[i][j]+1)
plt.matshow(squared_pred_arr)
plt.xlabel('Length of Real Answer')
plt.ylabel('Length of Predicted Answer')
plt.colorbar()
plt.show()

pred_real_ratio = [len(pred_answers[i])/len(real_answers[i]) if len(real_answers[i]) else 0
                   for i in range(len(pred_answers))]
print(sum(pred_real_ratio)/len(pred_real_ratio))

from collections import Counter
edit_dists_counter = Counter(edit_dists)
dists_list=list(edit_dists_counter.items()); dists_list.sort()
dists_keys = [i[0] for i in dists_list]
dists_vals = [i[1] for i in dists_list]

print('Average:',sum(edit_dists)/len(edit_dists))

tmp=0; dists_sum = sum(dists_vals)
for i, j in dists_list:
  tmp+=j
  print(f'편집 거리 {i} - {j}개 ({j/dists_sum*100:.2f}%) - 누적 {tmp/dists_sum*100:.2f}%')

plt.plot(dists_keys, dists_vals)
plt.xlabel('Edit Distance')
plt.ylabel('Number of Answers')
plt.show()

start_visualize = []
end_visualize = []

with torch.no_grad(), open(f'{args.NAME}.csv', 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])

    rows = []
    c = 0
    # for sample in tqdm(test_dataset, "Testing"):
    for sample in tqdm(indexed_test_dataset, "Testing"):
        input_ids, token_type_ids = [torch.tensor(sample[key], dtype=torch.long, device="cuda") for key in ("input_ids", "token_type_ids")]
        # print(sample)
    
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

        start_logits = output.start_logits
        end_logits = output.end_logits
        start_logits.squeeze_(0), end_logits.squeeze_(0)

        start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
        end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

        probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

        # 토큰 길이 8까지만
        for row in range(len(start_prob) - 8):
            probability[row] = torch.cat((probability[row][:8+row].cpu(), torch.Tensor([0] * (len(start_prob)-(8+row))).cpu()), 0)

        index = torch.argmax(probability).item()

        start = index // len(end_prob)
        end = index % len(end_prob)
        
        # 확률 너무 낮으면 자르기
        if start_prob[start] > 0.3 and end_prob[end] > 0.3:
            start_str = sample['position'][start][0]
            end_str = sample['position'][end][1]
        else:
            start_str = 0
            end_str = 0

        start_visualize.append((list(start_prob.cpu()), (start, end), (start_str, end_str)))
        end_visualize.append((list(end_prob.cpu()), (start, end), (start_str, end_str)))
        
        rows.append([sample["guid"], sample['context'][start_str:end_str]])

    writer.writerows(rows)





















