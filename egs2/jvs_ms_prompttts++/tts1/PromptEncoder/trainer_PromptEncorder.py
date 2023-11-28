import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.modeling_outputs import ModelOutput
from transformers.configuration_utils import PretrainedConfig
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import datetime

"""
参考サイト

BERTファインチューニング：
Transformer Trainerクラス：https://qiita.com/m__k/items/2c4e476d7ac81a3a44af

"""

d_today_utc = datetime.datetime.utcnow().date()

# input_data = '/mnt/data/users/snegishi/M2/espnet/egs2/jvs_ms_negishi/tts1/Xvector_Bert/data/NL_Xvector.csv'
input_data = '/mnt/data/users/snegishi/M2/espnet/egs2/jvs_ms_negishi/tts1/Xvector_Bert/data/full_NL_Xvector.csv'
model_output = f'./output/model_{d_today_utc}'
config_output = f'./output/model_{d_today_utc}'

#----------------------------------------------------------------------------------------------------
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

#データ準備-------------------------------------------------------------------------------------------
df = pd.read_csv(input_data,header=None)
df['list'] = df[df.columns[1:]].values.tolist()
df['comment_text'] = df[df.columns[:1]]
new_df = df[['comment_text', 'list']].copy()
vector_size = len(new_df['list'][0])

#データ分割～Datasetクラス(CustomDataset)を用意-------------------------------------------------------------------------------------------
# MAX_LEN = 200　#現在のプログラムでは使ってない
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1000
LEARNING_RATE = 1e-05
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-japanese') # <<< ここも bert-base-japanese
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

"""
#Trainerクラス用のDataset作成の都合で以下のCustomDatasetは使えなさそう
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
"""

#データ分割～Datasetクラス(BERTDataset)を用意-------------------------------------------------------------------------------------------
class BERTDataset(Dataset):
    def __init__(self, df):
        self.features = [
            {
                'comment_text': row.comment_text,
                'list': row.list
            } for row in tqdm(df.itertuples(), total=df.shape[0])
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = BERTDataset(train_dataset)
testing_set = BERTDataset(test_dataset)


#DataCollatorの定義-------------------------------------------------------------------------------------------
class BERTCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        examples = {
            'comment_text': list(map(lambda x: x['comment_text'], examples)),
            'list': list(map(lambda x: x['list'], examples))
        }
        
        encodings = self.tokenizer(examples['comment_text'],
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length,
                                   return_tensors='pt')
        encodings['list'] = torch.tensor(examples['list'])
        return encodings
    
bert_collator = BERTCollator(tokenizer)


#DataCollatorの動作確認-------------------------------------------------------------------------------------------
loader = DataLoader(training_set, collate_fn=bert_collator, batch_size=8, shuffle=True)

batch = next(iter(loader))
for k,v in batch.items():
    print(k, v.shape)
# input_ids torch.Size([8, 41])
# token_type_ids torch.Size([8, 41])
# attention_mask torch.Size([8, 41])
# category_id torch.Size([8])

print(batch)
# {'input_ids': tensor([[    2,  9680, 10520, 28770, 28865,   450,    52,    53,   512,  9594,
#           5359,   126,   243, 28673,    12,     6,  5359,    40, 16329, 28476,
#           2935,    63,  7388,   104,     6,   331, 28483,  4658,    35, 15288,・・・
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,・・・
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ・・・
# 'category_id': tensor([0, 8, 8, 2, 1, 7, 0, 2])}


#モデルの定義-------------------------------------------------------------------------------------------
#Trainerクラス用に諸々変更
class BERTClass(torch.nn.Module):
    def __init__(self, pretrained_model, num_labels ,loss_function=None,):
        super(BERTClass, self).__init__()
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-japanese') # <<<
        self.l1 = pretrained_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 512) #<<<
        self.loss_function = loss_function
        self.config = pretrained_model.config
        
        self.config.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None, output_attentions=False, output_hidden_states=False, list=None):
        output_1= self.l1(input_ids, attention_mask = attention_mask, position_ids = position_ids, token_type_ids = token_type_ids, output_attentions = output_attentions, output_hidden_states = output_hidden_states)
        output_2 = self.l2(output_1.last_hidden_state[:,0,:])
        output = self.l3(output_2)

        loss = None
        if list is not None and self.loss_function is not None:
            loss = self.loss_function(output, list)

        attentions=None
        if output_attentions:
            attentions=output_1.attentions

        hidden_states=None
        if output_hidden_states:
            hidden_states=output_1.hidden_states

        return ModelOutput(
            logits = output,
            loss = loss,
            last_hidden_state=output_1.last_hidden_state,
            attentions=attentions,
            hidden_states=hidden_states
        )

loss_fct = torch.nn.MSELoss()
pretrained_model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = BERTClass(pretrained_model, num_labels=vector_size ,loss_function=loss_fct)


#TrainingArgumentsの設定-------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir= model_output,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    label_names=['list'],
    lr_scheduler_type='constant',
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    remove_unused_columns=False,
    report_to='none'
)


#Trainerクラスの定義と実行-------------------------------------------------------------------------------------------
# !!!!!!!!!!実験用にtrainとtestを同じデータで過学習させる!!!!!!!!!!!!!!!!!!
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=bert_collator,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'])

trainer.save_model()
trainer.save_state()

config = model.config
config.num_labels = vector_size
config.save_pretrained(config_output)
