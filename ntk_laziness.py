# -*- coding: utf-8 -*-
"""ntk_laziness.ipynb
"""


!pip install nbconvert
!pip install transformers
!pip install datasets
import torch
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
from functorch import jacrev, make_functional_with_buffers
import gc
from torch import nn
from torch.nn.functional import relu
from torch.autograd import grad

import numpy as np
import torch
from datasets import load_dataset#, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import Dataset
import logging

from datasets import load_dataset

# Commented out IPython magic to ensure Python compatibility.
# %ls

# Commented out IPython magic to ensure Python compatibility.


raw_datasets  = load_dataset("glue", 'sst2')

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, AutoModelForSequenceClassification
#from roberta import RobertaForSequenceClassification


model_name = "bert-base-uncased"
#model_name = prajjwal1/bert-mini
#config.num_labels=2
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoTokenizer, DataCollatorWithPadding
def preprocessing_function(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=128)


tokenized_datasets = raw_datasets.map(preprocessing_function, batched=True)
# llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
tokenized_datasets.set_format("torch")

# Data collator for padding a batch of examples to the maximum length seen in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification
from transformers.activations import ACT2FN
import random
# %cd PEFT
# %ls
import leader
device = 'cuda' if torch.cuda.is_available() else 'cpu'



model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# %ls

# Commented out IPython magic to ensure Python compatibility.
# %cd PEFT


leader.PEFT(model, method='row', targets=['classifier'], rank=1)
#targets=['key', 'value', 'dense', 'query'])
# method = 'row', 'column', 'random'

#import evaluate
import numpy as np
from sklearn import metrics
import torch
import numpy as np

def compute_metrics(eval_pred):


    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)

    #precision = metrics.precision_score(labels, predictions, average="macro")
    #recall = metrics.recall_score(labels, predictions, average="macro")
    #f1 = metrics.f1_score(labels, predictions, average="macro")
    accuracy = metrics.accuracy_score(labels, predictions)
    return {'accuracy': accuracy}
    #return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

from transformers import TrainingArguments, Trainer

import time
from transformers import Trainer, TrainingArguments

tokenized_datasets["validation"]['sentence'][0:10]



def Kernal_Whole(tensor):
    # Get the dimensions of the input tensor
    N, M, P, Q = tensor.shape


      #a=[tensor[:,:,0,0], tensor[:,:,0,1]]
      #b=[tensor[:,:,1,0], tensor[:,:,1,1]]
      #torch.stack(a,b,dim=0)

    return result_tensor

# Function to tokenize input text
def tokenize_input(texts, tokenizer, max_length=10):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Example input sets
ntk_sample_size=4
texts_set_1 = tokenized_datasets["validation"]['sentence'][0:ntk_sample_size]
texts_set_2 = tokenized_datasets["validation"]['sentence'][ntk_sample_size:2*ntk_sample_size]
print(f' NTK sampe size is {ntk_sample_size}')

# Tokenize inputs
input_set_1 = tokenize_input(texts_set_1, tokenizer)
input_set_2 = tokenize_input(texts_set_2, tokenizer)

# Move inputs to the same device as the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_set_1 = {k: v.to(device) for k, v in input_set_1.items()}
input_set_2 = {k: v.to(device) for k, v in input_set_2.items()}
model = model.to(device)

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn as nn
from functorch import make_functional_with_buffers, vmap, jacrev


# Get word embeddings instead of text input
def get_word_embeddings(input_ids):
    with torch.no_grad():
        outputs = model.bert.embeddings(input_ids)
    return outputs

# We will focus on the last layer output
class BertLastLayer(nn.Module):
    def __init__(self, bert_model):
        super(BertLastLayer, self).__init__()
        self.bert_model = bert_model

    def forward(self, embeddings):
        # Get the last hidden states from BERT
        outputs = self.bert_model(inputs_embeds=embeddings)
        # We are only interested in the last hidden state
        last_hidden_state = outputs.logits
        return last_hidden_state  # CLS token's representation for simplicity

# Example usage:

bert_last_layer = BertLastLayer(model).to(device)

# Example sentence
sentence = ["hello", 'world']

# Tokenize the sentence
#inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
input_ids_train = input_set_1['input_ids']

# Get word embeddings for the sentence
x_train = get_word_embeddings(input_ids_train)

input_ids_test = input_set_2['input_ids']

# Get word embeddings for the sentence
x_test = get_word_embeddings(input_ids_test)

# Convert the BERT model to a functional model using functorch, including buffers
fnet, params, buffers = make_functional_with_buffers(bert_last_layer)

# Function for a single pass through the functional model
def fnet_single(params, buffers, x):
    return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

# NTK Calculation (similar to your original code)
def empirical_ntk_jacobian_contraction(fnet_single, params, buffers, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, None, 0))(params, buffers, x1)
    jac1 = [j.flatten(2) for j in jac1]
    print('params',len(params))
    print('buffers',len(buffers))

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, None, 0))(params, buffers, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

# Compute NTK
result_from_ntk_vps = empirical_ntk_jacobian_contraction(fnet_single, params, buffers, x_train, x_test,compute='full')
print(result_from_ntk_vps.shape)
ntk_result = Kernal_Whole(result_from_ntk_vps)
print(ntk_result.shape)

import matplotlib.pyplot as plt
import seaborn as sns


result_ntk_np_pre = ntk_result.detach().cpu().numpy()  # Convert to CPU and NumPy

# Use seaborn's heatmap function to plot
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(result_ntk_np_pre, cmap="viridis")  # You can change the color map to 'coolwarm', 'magma', etc.
plt.title("Heatmap of NTK")
plt.xlabel("Test Samples")
plt.ylabel("Train Samples")
plt.title('Pretrain')

plt.savefig('NTK_pretrain.jpg', format='jpg', dpi=300)
print("----pretain NTK done---------")
plt.show()

import pickle
import numpy as np

# Save the array to a file
with open('pretrain.pkl', 'wb') as f:
    pickle.dump(result_ntk_np_pre, f)

# Count of trainable parameters
total_trainable_params = 0
total =  0
# Print trainable parameters and count their total number
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}, Shape: {param.shape}")

        total_trainable_params += param.numel()
    total+=param.numel()

print(f"Total trainable parameters:{total_trainable_params}, percentage:  {total_trainable_params/total}")

#!pwd

#trainer.save_model("dir")

#train one epoch

num_samples = len(tokenized_datasets["train"])
steps_per_epoch = num_samples // 32  # Assuming batch size is 32
max_steps_half_epoch = steps_per_epoch // 2
max_steps=max_steps_half_epoch

training_args = TrainingArguments(
    output_dir='dir',
    learning_rate=2e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.00,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    save_steps=10000000,
    logging_steps=100,

    load_best_model_at_end=True,
    lr_scheduler_type="cosine",  # You can choose from 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', etc.
    warmup_steps=100,
    #max_steps=max_steps_half_epoch  # Train for half epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],

    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

#plot heat map
result_from_ntk_vps = empirical_ntk_jacobian_contraction(fnet_single, params, buffers, x_train, x_test,compute='full')
ntk_result = Kernal_Whole(result_from_ntk_vps)

result_ntk_np = ntk_result.detach().cpu().numpy()  # Convert to CPU and NumPy

import matplotlib.pyplot as plt
import seaborn as sns


# Use seaborn's heatmap function to plot
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(result_ntk_np, cmap="viridis")  # You can change the color map to 'coolwarm', 'magma', etc.
plt.title("Heatmap of NTK")
plt.xlabel("Test Samples")
plt.ylabel("Train Samples")
plt.title('Epoch 1')
plt.savefig('NTK_one_epoch.jpg', format='jpg', dpi=300)
print("----one epoch NTK done---------")
plt.show()

type(result_ntk_np)

#!huggingface-cli scan-cache

#!huggingface-cli delete-cache

#!pip show transformers

#!pip install seaborn

#!huggingface-cli cache list

#!huggingface-cli cache delete

#!ls ~/.cache/huggingface

#!echo $HF_HOME

#!rm -rf ~/.cache/huggingface

result_ntk_np_pre.shape, result_ntk_np.shape

import numpy as np
import pickle
import numpy as np
# Load the array from the file
with open('pretrain.pkl', 'rb') as f:
    result_ntk_np_pre = pickle.load(f)

with open('leader.pkl', 'rb') as f:
    result_ntk_np = pickle.load(f)

# Calculate the Frobenius norm (distance between the two matrices)
frobenius_norm = np.linalg.norm(result_ntk_np_pre - result_ntk_np, 'fro')/np.linalg.norm(result_ntk_np_pre, 'fro')

frobenius_norm

import matplotlib.pyplot as plt
import seaborn as sns


# Use seaborn's heatmap function to plot
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(result_ntk_np, cmap="viridis")  # You can change the color map to 'coolwarm', 'magma', etc.
plt.title("Heatmap of NTK")
plt.xlabel("Test Samples")
plt.ylabel("Train Samples")
plt.title('Epoch 1')
plt.savefig('NTK_one_epoch.jpg', format='jpg', dpi=300)
print("----one epoch NTK done---------")
plt.show()

# Use seaborn's heatmap function to plot
plt.figure(figsize=(10, 8))  # Set the size of the figure
sns.heatmap(result_ntk_np_pre, cmap="viridis")  # You can change the color map to 'coolwarm', 'magma', etc.
plt.title("Heatmap of NTK")
plt.xlabel("Test Samples")
plt.ylabel("Train Samples")
plt.title('Pretrain')

plt.savefig('NTK_pretrain.jpg', format='jpg', dpi=300)
print("----pretain NTK done---------")



plt.show()

#Perform the following  iterations for different  models  and tasks   bert/Roberta sst2/Glue to plot their laziness

frobenius_norms = []
for  ntk_step in  range(10):
   result_from_ntk_vps = empirical_ntk_jacobian_contraction(fnet_single, params, buffers, x_train, x_test,compute='full')
   ntk_result = Kernal_Whole(result_from_ntk_vps)
   result_ntk_np = ntk_result.detach().cpu().numpy()  # Convert to CPU and NumPy
   #Dump NTK of each step to .pkl
   plt.figure(figsize=(10, 8))  # Set the size of the figure
   sns.heatmap(result_ntk_np, cmap="viridis")  # You can change the color map to 'coolwarm', 'magma', etc.
   plt.title("Heatmap of NTK")
   plt.xlabel("Test Samples")
   plt.ylabel("Train Samples")
   plt.title(f'Epoch {ntk_step}')
   plt.savefig(f'NTK_epoch_{ntk_step}.jpg', format='jpg', dpi=300)
   print(f"---NTK epoch {ntk_step} done---------")

   if ntk_step == 0:
        # Preserve the result for the first step
        result_ntk_0 = result_ntk_np.copy()
        fro_0=np.linalg.norm(result_ntk_0, 'fro')

   if ntk_step > 0:
        frobenius_norm = np.linalg.norm(result_ntk_np - result_ntk_0, 'fro')/fro_0
        frobenius_norms.append(frobenius_norm)
        print('current frob norm',frobenius_norm)

   trainer.train()

# Plot Frobenius norms
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(frobenius_norms) + 1), frobenius_norms, marker='o', linestyle='-', color='b')
plt.title("Frobenius Norm of NTK Difference")
plt.xlabel("Iteration Number")
plt.ylabel("Frobenius Norm")
plt.grid(True)
plt.savefig('Frobenius_norm_plot.jpg', format='jpg', dpi=300)
plt.show()


#Eigen value decopomsition

# loRa
# get NTK of LoRa
# eigen value of LoRa NTK

# Row PEFT

# Column PEFT

# Random


# are the Eigen-value  spectrun of three methods similar to LoRa?
