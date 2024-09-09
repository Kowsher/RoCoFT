# sep 7
# modified to update only rows/columns 
# sep 6 
# wide network simulation 

import torch 
import torch.nn as nn 
import torch.optim as optim

from leader import row, column, random, PEFT 


input_dim = 3 
hidden_width = 500 



class WideNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits




class IterDataset(torch.utils.data.IterableDataset):
    #def __init__(self, generator):
        #self.generator = generator
    def __init__(self):
        pass

    def __iter__(self):
        #return self.generator
        while True:
            x = torch.randint(2, (input_dim,))
            #y = torch.logical_xor(x).long()
            y = torch.remainder(torch.sum(x), 2)
            x = x.to(torch.float)
            y = y.long()
            yield x,y



model = WideNetwork()

# peft modification
PEFT(model, row, 1)


model.train()
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1E-2)


my_dataset = IterDataset()
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size = 32)


for batch_idx, (x, y) in enumerate(train_loader):
    optimizer.zero_grad()
    pred = model(x)
    # this would be needed for regression MSE loss 
    #y = torch.unsqueeze(y,1)

    my_loss = loss(pred, y)

    my_loss.backward()
    optimizer.step()

    print(my_loss.item())


    if batch_idx>10000:
        break 












