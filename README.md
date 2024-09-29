# RoCoFT

This is the implementation for the paper "RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates".

We explore a set of novel PEFT methods where the fine-tuning is applied on rows and columns of weights of the model.
In RoCoFT$_\text{row}$ we finetune the selected weight $\mathbf{W}$ as $$\mathbf{W}=\mathbf{W}+\mathbf{R}$$ and in RoCoFT$_\text{col}$ we finetune the selected weight $\mathbf{W}$ as $\mathbf{W}=\mathbf{W}+\mathbf{C}$$ 
We  demonstrate  the higher  memroy and time effivency of our method  in comparison with other PEFT methods.
This performance is governed by the less number of parameters selected for  fine-tuning.
Through numerical experiments, we demonstrate that the accuracy of  our method is comparable to other PEFT methods.
