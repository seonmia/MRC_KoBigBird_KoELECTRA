#!/usr/bin/env python
# coding: utf-8

# # Colab Mount & Import Libraries

# In[ ]:


pwd


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


train_file = "data/train.json"
add_train_file = 'data/017.뉴스 기사 기계독해 데이터/01.데이터/1.Training/라벨링데이터/TL_span_extraction.json'
add_train_rev_file = "data/data_extraction_answer_len20.json"
test_file = "data/test.json"
blank_file = "data/blank.csv"


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/GoormProject/GoormProject2')


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install wandb')


# In[ ]:





# In[ ]:


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


# In[ ]:





# # Config

# In[ ]:


args = edict({'w_project': 'KoBigBird',
              'w_entity': 'ushape', # WandB ID
              'learning_rate': 2e-4,
              'batch_size': {'train': 128,
                             'eval': 4,
                             'test': 128},
              'accumulate': 32,
              'epochs': 10,
              'seed': 42,
              'model_name': 'monologg/kobigbird-bert-base',
              'max_length': 2048})
args['NAME'] = ''f'kobigbird_v2_ep{args.epochs}_max{args.max_length}_lr{args.learning_rate}_{random.randrange(0, 1024)}'
print(args.NAME)


# # Setting Model

# In[ ]:


def seed_everything(seed):
   random.seed(seed)
   np.random.seed(seed)
   os.environ["PYTHONHASHSEED"] = str(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)  # type: ignore
   torch.backends.cudnn.deterministic = True  # type: ignore
   torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(args.seed)


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(args.model_name)


# In[ ]:


# model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
# model.to(device)


# In[ ]:


model.cuda();


# In[ ]:


optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


# # Preprocessing

# ## KoMRC

# In[ ]:


from typing import List, Tuple, Dict, Any
import json
import random

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


# In[ ]:


from pprint import pprint
dataset = KoMRC.load(train_file)
pprint(dataset[0])


# ## TokenizedKoMRC

# In[ ]:


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
        
        # 토크나이저 변경에 따른 ## 대응 및 포지션 조정
        for morph in sentence_tokens:
            if len(morph) > 2:
                if morph[:2] == '##':
                    morph = morph[2:]

            position = sentence.find(morph, position)
            tokens.append((morph, (position, position + len(morph)))) 
            position += len(morph) 
        
        return tokens
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)

        context, position = zip(*self._tokenize_with_position(sample['context']))
        context, position = list(context), list(position)

        question = self._tokenizer.tokenize(sample['question'])

        if sample['answers'] is not None:
            answers = []

            # 여러 답 중 짧은 답만 학습
            local_shortest_answer_len = 200; local_shortest_answer_idx = 0
            for i, answer in enumerate(sample['answers']):
                if len(answer['text'])<local_shortest_answer_len:
                  local_shortest_answer_len = len(answer['text'])
                  local_shortest_answer_idx=i
            answer=sample['answers'][local_shortest_answer_idx]
            for start, (position_start, position_end) in enumerate(position):
                if position_start <= answer['answer_start'] < position_end:
                    break

            target = ''.join(answer['text'].split(' '))
            source = ''
            for end, morph in enumerate(context[start:], start):
                source += morph
                if target in source:
                    break

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


# In[ ]:


dataset = TokenizedKoMRC.load(train_file)
train_dataset, dev_dataset = TokenizedKoMRC.split(dataset)
print("Number of Samples:", len(dataset))
print("Number of Train Samples:", len(train_dataset))
print("Number of Dev Samples:", len(dev_dataset))
print(dev_dataset[0])


# ## Indexer

# ### current

# In[ ]:


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


# In[ ]:


indexer = Indexer(list(tokenizer.vocab.keys()))
print(indexer.sample2ids(dev_dataset[0]))


# ## IndexerWrappedDataset

# In[ ]:


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


# In[ ]:


indexed_train_dataset = IndexerWrappedDataset(train_dataset, indexer)
indexed_dev_dataset = IndexerWrappedDataset(dev_dataset, indexer)

sample = indexed_dev_dataset[0]
for i in sample:
  print(i,sample[i])


# ## Collator

# In[ ]:


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


# In[ ]:


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


# In[ ]:


batch = next(iter(dev_loader))
print(batch['input_ids'].shape)
print(batch['input_ids'])
print(list(batch.keys()))


# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ## Edit Distance

# In[ ]:


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


# # Test

# ## 데이터셋 불러오기

# In[ ]:


test_dataset = TokenizedKoMRC.load(test_file)
indexer_test = Indexer(list(tokenizer.vocab.keys()))
indexed_test_dataset = IndexerWrappedDataset(test_dataset, indexer_test)
print("Number of Test Samples", len(test_dataset))


# In[ ]:


print(best_model[0])
print(f'models/{args.NAME}_{best_model[0]}')


# In[ ]:


저장된 베스트 모델 가져오기 
model = AutoModelForQuestionAnswering.from_pretrained(f'models/{args.NAME}_{best_model[0]}')
model.cuda();


# In[ ]:


for i in indexed_train_dataset[1]:
  print(i,':',indexed_train_dataset[1][i])


# In[ ]:


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


# ## 분석용 함수

# In[ ]:


def get_pred_answer(dataset, prob_threshold=0.0, token_len_threshold=None, len_threshold=None):
  probs = []
  pred_answers = []
  real_answers = []
  for sample in tqdm(dataset):
      
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

      # 토큰 길이를 token_len_threshold까지만
      if token_len_threshold:
        for row in range(len(start_prob) - token_len_threshold):
            probability[row] = torch.cat((probability[row][:token_len_threshold+row].cpu(), torch.Tensor([0] * (len(start_prob)-(token_len_threshold+row))).cpu()), 0)

      index = torch.argmax(probability).item()
      
      start = index // len(end_prob)
      end = index % len(end_prob)

      # 확률이 prob_threshold 이하이면 자르기
      start_str = sample['position'][start][0]
      end_str = sample['position'][end][1]
      #if start_prob[start] > prob_threshold and end_prob[end] > prob_threshold:
      #    end_str = sample['position'][end][1]
      #else:
      #    end_str = min(start_str+len_threshold,sample['position'][end][1])

      pred_ans = sample['context'][start_str:end_str]

      #edit_dists.append(edit_distance(pred_ans,sample['answers_text']))
      probs.append(min(start_prob[start], end_prob[end]))
      pred_answers.append(pred_ans); real_answers.append(sample['answers_text'])
  return pred_answers, real_answers, probs


# In[ ]:


from collections import Counter

def anal_edit_dists(edit_dists, print_log=True, text='', write_log=[]):
  edit_dists_counter = Counter(edit_dists)
  dists_list=list(edit_dists_counter.items()); dists_list.sort()
  dists_keys = [i[0] for i in dists_list]
  dists_vals = [i[1] for i in dists_list]
  dists_l = []
  for i in edit_dists_counter:
    dists_l+=[i]*edit_dists_counter[i] 

  print('평균:',np.mean(dists_l))
  print('표준편차:', np.std(dists_l))
  write_log.append((np.mean(dists_l),np.std(dists_l)))
  if print_log:
    tmp=0; dists_sum = sum(dists_vals)
    for i, j in dists_list:
      tmp+=j
      print(f'편집 거리 {i} - {j}개 ({j/dists_sum*100:.2f}%) - 누적 {tmp/dists_sum*100:.2f}%')
  plt.plot(dists_keys, dists_vals)
  plt.xlabel('Edit Distance'+text)
  plt.ylabel('Number of Answers')
  plt.show()


# In[ ]:


def heatmap(pred_answers, real_answers, do_log=False, size=20, text=''):
  pred_arr = [[0]*(size+1) for _ in range(size+1)]
  for i in range(len(pred_answers)):
    if len(real_answers[i])>size or len(pred_answers[i])>size: continue
    pred_arr[len(real_answers[i])][len(pred_answers[i])]+=1
  #for i in pred_arr:
  #  print(i)
  if do_log:
    for i in range(size+1):
      for j in range(size+1):
        pred_arr[i][j] = math.log(pred_arr[i][j]+1)
  plt.matshow(pred_arr)
  plt.xlabel('Length of Real Answer'+text)
  plt.ylabel('Length of Predicted Answer')
  plt.colorbar()
  plt.show()


# ## Test Data Prediction

# In[ ]:


prob_threshold = 0.05
len_threshold = 9
token_len_threshold = 8


# In[ ]:


start_visualize = []
end_visualize = []

with torch.no_grad(), open(f'{best_model_name}_0803.csv', 'w') as fd:
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

      # 토큰 길이를 token_len_threshold까지만
        for row in range(len(start_prob) - token_len_threshold):
            probability[row] = torch.cat((probability[row][:token_len_threshold+row].cpu(), torch.Tensor([0] * (len(start_prob)-(token_len_threshold+row))).cpu()), 0)

        index = torch.argmax(probability).item()

        start = index // len(end_prob)
        end = index % len(end_prob)
        
      # 확률이 prob_threshold 이하이면 자르기
        start_str = sample['position'][start][0]
        if start_prob[start] > prob_threshold and end_prob[end] > prob_threshold:
            end_str = sample['position'][end][1]
        else:
            end_str = min(sample['position'][end][1], start_str+len_threshold)

        start_visualize.append((list(start_prob.cpu()), (start, end), (start_str, end_str)))
        end_visualize.append((list(end_prob.cpu()), (start, end), (start_str, end_str)))
        
        rows.append([sample["guid"], sample['context'][start_str:end_str]])

    writer.writerows(rows)


# In[ ]:


from collections import Counter
row_len = Counter([len(i[1]) for i in rows])


# In[ ]:


row_len


# In[ ]:


tmp_nums=[1,23,493,695,1023,1954,2069,3048,3549]
for n in tmp_nums:
#for n in range(len(rows)):
  #if len(rows[n][1])<19: continue
  for i in range(0,len(indexed_test_dataset[n]['context']),100):
    print(indexed_test_dataset[n]['context'][i:i+100])
  print('Q:',indexed_test_dataset[n]['question'])
  print('A:',rows[n][1])
  print()


# In[ ]:


indexed_test_dataset[0].keys()


# # Self Test

# In[ ]:


prob_threshold=0.05
second_prob_threshold=0.25
token_len_threshold=8
len_threshold=5
second_len_threshold=5


# In[ ]:


pred, real, prob = get_pred_answer(indexed_dev_dataset,prob_threshold=prob_threshold,token_len_threshold=token_len_threshold,len_threshold=len_threshold)


# In[ ]:


edit = []
for i in range(len(pred)):
  pr = pred[i]
  if prob[i]<=prob_threshold: pr=pr[:len_threshold]
  #elif prob[i]<=second_prob_threshold: pr=pr[:second_len_threshold]
  ed = edit_distance(pr, real[i])
  edit.append(ed)


# In[ ]:


edit_prob=[[] for _ in range(10)]
for i in range(len(real)):
  #edit_prob[int(prob[i]*10)].append(edit[i])
  edit_prob[int(prob[i]*10)].append((len(pred[i])-len(real[i])))
for p, i in enumerate(edit_prob):
  print(p,':',np.mean(i), len(i),i)


# In[ ]:


x_label=[i*0.1 for i in range(10)]
plt.plot(x_label,list(np.mean(i) for i in edit_prob))
plt.rcParams.update({'font.size': 12})
plt.xlabel('Probability of KoBigBird')
plt.ylabel('Average len(predict) - len(real)')
plt.show()


# In[ ]:


x_label=[i*0.1 for i in range(10)]
plt.plot(x_label,list(len(i) for i in edit_prob))
plt.xlabel('Probability of KoBigBird')
plt.ylabel('Number of Samples')
plt.show()


# In[ ]:


anal_edit_dists(edit, print_log=False,text=f' (prob=0.25)')


# In[ ]:


heatmap(pred,real,do_log=True,text=f' (prob=0.25)')


# In[ ]:


from collections import defaultdict
edit_counter = {i:defaultdict(int) for i in range(70)}
for i in range(len(pred)):
  if edit[i]>=70: continue
  edit_counter[edit[i]][len(real[i])-len(pred[i])]+=1


# In[ ]:


#for i in edit_counter:
#  if i>30: break
#  print(f'{i}: {edit_counter[i]}')


# In[ ]:


import math
edit_show = [[0]*17 for _ in range(21)]
for i in edit_counter:
  if i>20: break
  for j in edit_counter[i]:
      try: edit_show[i][8+int(8*(j/i if i else 0))]+=edit_counter[i][j]
      except: edit_show[i][16]+=edit_counter[i][j]
for i in edit_show:
  for j in range(len(i)):
    i[j] = math.log(i[j]+1)


# In[ ]:


import matplotlib.pyplot as plt

plt.matshow(edit_show)
plt.xlabel('(len(Predicted Answer) - len(Real Answer)) / Edit Distance')
plt.ylabel(f'Edit Distance (prob={prob_threshold})')
plt.xticks([0,8,16],labels=['-1','0','1'])
plt.show()


# In[ ]:


def eval_edit(prob_threshold, max_len):
  edit_dists=[]
  for i in zip(pred,real,prob):
    ed = edit_distance(i[0],i[1]) if i[2]>prob_threshold else edit_distance(i[0][:max_len],i[1])
    edit_dists.append(ed)
  return np.mean(edit_dists), np.std(edit_dists)


# In[ ]:


width=12
edit_len_prob = [[0]*width for _ in range(10)]; edit_len_prob_std = [[0]*width for _ in range(10)]
# answer len: 0 to 9
# prob threshold: 0.00 to 0.45
for local_prob in range(10):
  for local_len in range(width):
    m,s = eval_edit(local_prob*0.05, local_len)
    edit_len_prob[local_prob][local_len]=m
    edit_len_prob_std[local_prob][local_len]=s


# In[ ]:


import matplotlib.pyplot as plt

fig,ax=plt.subplots()
ax.imshow(edit_len_prob)
plt.ylabel('Probability Threshold')
plt.xlabel(f'Max len when (prob < prob_threshold)')
plt.yticks([0,2,4,6,8],labels=[0,0.1,0.2,0.3,0.4])
plt.figure(figsize=(15,20))
#plt.rcParams.update({'font.size': 7})
for i in range(10):
  for j in range(width):
      ax.text(j,i,round(edit_len_prob[i][j],3), ha='center',va='center',color='w', fontsize=7)
plt.show()


# In[ ]:


fig,ax=plt.subplots()
ax.imshow(edit_len_prob_std)
plt.ylabel('Probability Threshold')
plt.xlabel(f'Max len when (prob < prob_threshold)')
plt.yticks([0,2,4,6,8],labels=[0,0.1,0.2,0.3,0.4])
plt.figure(figsize=(15,20))
#plt.rcParams.update({'font.size': 7})
for i in range(10):
  for j in range(width):
      ax.text(j,i,round(edit_len_prob_std[i][j],3), ha='center',va='center',color='w', fontsize=7)
plt.show()


# In[ ]:


'''count=0
for i in range(len(real)):
  #if len(pred[i])>=len(real[i])*2:
  #if len(real[i])>15:
  print(f'#{i}:',indexed_dev_dataset[i]['question'])
  print('Real:',real[i])
  print('Pred:',pred[i])
  print('Edit Distance:',edit[i])
  count+=1
  if count>20: break'''


# # 실험

# ## Validation Data 실험 - 길이

# In[ ]:


# 아무런 제한 없이
# get_pred_answer(dataset, prob_threshold=0.0, token_len_threshold=None)
pred_1, real, edit_1 = get_pred_answer(indexed_dev_dataset)


# In[ ]:


anal_edit_dists(edit_1, print_log=False,text=f' (prob=0.0)')


# In[ ]:


print(len(pred_1),len(real))
heatmap(pred_1,real,do_log=True,text=f' (prob=0.0)')


# In[ ]:


'''pred=[[] for _ in range(11)];edit=[[] for _ in range(11)]
for l in range(1,11):
  print(f'Token MAX length = {l}')
  pred[l], _, edit[l] = get_pred_answer(indexed_dev_dataset, token_len_threshold=l)
  anal_edit_dists(edit[l], print_log=False,text=f' (token_max_len={l})')
  heatmap(pred[l],real,do_log=True,text=f' (token_max_len={l})')'''


# In[ ]:


pred+=[[] for _ in range(11)];edit+=[[] for _ in range(11)]
for l in range(11,21):
  print(f'Token MAX length = {l}')
  pred[l], _, edit[l] = get_pred_answer(indexed_dev_dataset, token_len_threshold=l)
  anal_edit_dists(edit[l], print_log=False,text=f' (token_max_len={l})')
  heatmap(pred[l],real,do_log=True,text=f' (token_max_len={l})')


# In[ ]:


token_len_log=[]
for l in range(1,21):
  #print(f'Probability = {prob*10}%')
  #anal_edit_dists(edit[prob], print_log=False,text=f' (prob={prob*0.1:.1f})')
  #heatmap(pred[prob],real,do_log=True,text=f' (prob={prob*0.1:.1f})')
  print(f'Token MAX length = {l}')
  anal_edit_dists(edit[l], print_log=False,text=f' (token_max_len={l})',write_log=token_len_log)
  heatmap(pred[l],real,do_log=True,text=f' (token_len={l})')


# In[ ]:


token_len_log


# ## Validation 실험 - 확률

# In[ ]:


'''pred_prob=[[] for _ in range(10)];edit_prob=[[] for _ in range(10)]
token_prob_log = []
for prob in range(1,10):
  print(f'Probability = {prob*10}%')
  pred_prob[prob], _, edit_prob[prob] = get_pred_answer(indexed_dev_dataset, prob_threshold=prob*0.1)
  anal_edit_dists(edit_prob[prob], print_log=False,text=f' (prob={prob*0.1:.1f})',write_log=token_prob_log)
  heatmap(pred_prob[prob],real,do_log=True,text=f' (prob={prob*0.1:.1f})')'''


# In[ ]:


token_prob_log


# ## Validation 실험 - 종합

# In[ ]:


# 매 조건마다 학습을 다시 돌리는 건 비효율적이라 내부에서 구한 확률을 따로 빼버림
def get_pred_answer_eff(dataset):
  pred_answers = []
  for sample in tqdm(dataset):
      
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

      pred_answers.append((start_prob, end_prob))
  return pred_answers


# In[ ]:


token_len_idx_cache={}


# In[ ]:


def get_pred(data_prob, dataset, prob_threshold=0.0, token_len_threshold=None):
  print(f'prob_threshold = {prob_threshold}, token_len_threshold = {token_len_threshold}')
  edit_dists = []
  #pred_answers = []
  #real_answers = []
  idx_cache = []
  
  for idx in tqdm(range(len(data_prob))):
      start_prob, end_prob = data_prob[idx]
      sample = dataset[idx]
      # 계산 시간 좀 줄일려고 일종의 DP를 씀
      if token_len_threshold in token_len_idx_cache: index = token_len_idx_cache[token_len_threshold][idx]
      else:
          probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

          # 토큰 길이를 token_len_threshold까지만
          if token_len_threshold:
            for row in range(len(start_prob) - token_len_threshold):
                probability[row] = torch.cat((probability[row][:token_len_threshold+row].cpu(), torch.Tensor([0] * (len(start_prob)-(token_len_threshold+row))).cpu()), 0)

          index = torch.argmax(probability).item()
          idx_cache.append(index)
      
      start = index // len(end_prob)
      end = index % len(end_prob)

      # 확률이 prob_threshold 이하이면 자르기
      if start_prob[start] > prob_threshold and end_prob[end] > prob_threshold:
          start_str = sample['position'][start][0]
          end_str = sample['position'][end][1]
      else:
          start_str = 0
          end_str = 0

      pred_ans = sample['context'][start_str:end_str]

      edit_dists.append(edit_distance(pred_ans,sample['answers_text']))
      #pred_answers.append(pred_ans); real_answers.append(sample['answers_text'])
  if token_len_threshold not in token_len_idx_cache: token_len_idx_cache[token_len_threshold] = idx_cache
  return np.mean(edit_dists), np.std(edit_dists)


# In[ ]:


#data_probs = get_pred_answer_eff(indexed_dev_dataset)


# In[ ]:


#pred_tmp, real_tmp, edit_tmp = get_pred(data_probs, indexed_dev_dataset)


# In[ ]:


complex_mean = [[0]*5 for _ in range(6)]
complex_std = [[0]*5 for _ in range(6)]
complex_mem = {}


# In[ ]:


'''i=0
for leng in range(5,16,2):
  j=0
  for prob in range(0,10,2):
    mean,std = get_pred(data_probs, indexed_dev_dataset,prob_threshold=prob*0.1, token_len_threshold=leng)
    complex_mean[i][j] = mean
    complex_std[i][j] = std
    complex_mem[(leng,prob)] = (mean,std)
    j+=1
  i+=1'''


# In[ ]:


i=5
complex_mean.append([0]*5)
complex_std.append([0]*5)
j=0
for prob in range(0,10,2):
  mean,std = get_pred(data_probs, indexed_dev_dataset,prob_threshold=prob*0.1, token_len_threshold=15)
  complex_mean[i][j] = mean
  complex_std[i][j] = std
  complex_mem[(leng,prob)] = (mean,std)
  j+=1


# In[ ]:


for i in complex_mean:
  for j in i:
    print(j, end='\t')
  print()


# In[ ]:


for i in complex_std:
  for j in i:
    print(j, end='\t')
  print()


# In[ ]:


complex_mem


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(complex_mean)

plt.xlabel('Minimum Probability')
plt.ylabel('Maximum Token Length')
ax.set_title("Average Edit Distance")
for i in range(6):
  for j in range(5):
    text = ax.text(j,i,round(complex_mean[i][j],2),ha='center',va='center',color='w')
#plt.xlim([5,16])
#plt.ylim([0,1])
ax.set_xticks(np.arange(5))
ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8])
ax.set_yticks(np.arange(6))
ax.set_yticklabels([5,7,9,11,13,15])
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.imshow(complex_std)

plt.xlabel('Minimum Probability')
plt.ylabel('Maximum Token Length')
ax.set_title("Standard Deviation of Edit Distance")
for i in range(6):
  for j in range(5):
    text = ax.text(j,i,round(complex_std[i][j],2),ha='center',va='center',color='w')
#plt.xlim([5,16])
#plt.ylim([0,1])
ax.set_xticks(np.arange(5))
ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8])
ax.set_yticks(np.arange(6))
ax.set_yticklabels([5,7,9,11,13,15])
plt.show()

