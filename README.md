# RoCoFT

This is the implementation for the paper "RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates".

We explore a set of novel PEFT methods where the fine-tuning is applied on rows and columns of weights of the model.
In RoCoFT-row we finetune the selected weight $\mathbf{W}$ as $\mathbf{W}=\mathbf{W}+\mathbf{R}$  and in RoCoFT-col we finetune the selected weight $\mathbf{W}$ as $\mathbf{W}=\mathbf{W}+\mathbf{C}$. 
We  demonstrate  the higher  memroy and time efficiency of our method  in comparison with other PEFT methods.
This performance is governed by the less number of parameters selected for fine-tuning and also similarity of the  spectrum of the kernels induced  by $\mathbf{R}$ and $\mathbf{C}$.
Through numerical experiments, we demonstrate that the accuracy of  our method is comparable to other PEFT methods.


## Requirements
Use python 3.11 from MiniConda

- torch==2.3.0
- accelerate==0.33.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.12.0
- transformers==4.44.0
- deepspeed==0.15.1
- sentencepiece==0.2.0


To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the datasets from hugginface

## Quick Demos

To get started with `propulsion`, follow these simple steps:

1. **Import the necessary modules:**

    ```python
    import leader
    from transformers import RobertaForSequenceClassification
    ```

2. **Load a pre-trained model and apply PEFT:**

    ```python
    model = RobertaForSequenceClassification.from_pretrained('model_name')
    leader.PEFT(model, method='row', rank=1) 
    ```

3. **For column**

    ```python
    leader.PEFT(model, method='column', rank=1) 
    ```
4. **In order to choose row or column using pruning technique**

    ```python
    import leader
    
    # Example text input
    input_text = tokenized_datasets['train']['sentence'][0:200]
    
    input_text = []
    for i in tqdm(range(len(tokenized_datasets['train']['premise'][0:200]))):
        st = tokenized_datasets['train']['premise'][i] + tokenized_datasets['train']['hypothesis'][i]
        input_text.append(st)
       
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    
    
    # Get the embedding of the input text
    with torch.no_grad():
        embeddings = model.roberta.embeddings(input_ids)

    leader.PEFT_prunig(model, method='column', rank=3, input_data=embeddings, descending=False)
    ```
If descending is true, pruning method return the least weights. 
