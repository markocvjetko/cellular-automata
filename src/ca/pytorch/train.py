import src.game_of_life.pytorch.models as models
import src.game_of_life.pytorch.data as data
import torch.nn as nn
import torch

'''A minimalistic neural network trained to mimic the Game of Life update rule.

    Currently, the outcome of training highly depends on the random initialization of weights.
    The model is not guaranteed to converge to the correct update rule. Correct kernel found 
    ~20% of the runs. Given the network size genetic algorithms or parameter space search 
    algorithms might be more robust approaches to finding the correct update rule.
'''

if __name__ == "__main__":

    dataset = data.GameOfLifeDataset((20, 20), p=0.4, n_samples=1000)

    model = models.GameOfLifeModel()
    model.train()
    model.cuda()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(30):
        for i, (inputs, labels) in enumerate(dataloader):

            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = loss_fn(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        #stoppping criterion
        if loss.item() < 0.02:
                print('outputs', outputs[0, 0, 0:3, 0:3].cpu().detach().numpy())
                print('target', labels[0, 0:3, 0:3].cpu().detach().numpy())

                print(model.state_dict())
                break
            
        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
