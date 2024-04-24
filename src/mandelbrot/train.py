import torch
import torch.nn as nn
from model import MandelbrotModel
from data import MandelbrotDataset
import matplotlib.pyplot as plt
import numpy as np

def main():

    model = MandelbrotModel()
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dataset = MandelbrotDataset(root='data/mandelbrot')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            input, label = data
            #swap dim 0 and 1
            input = input.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    #visuallize the model's prediction. Feed the first image in the dataset to the model and plot next 1000 images in the sequence    
    with torch.no_grad():
        input, _ = dataset[0]
        input = input.cuda()
        input = input.unsqueeze(0)
        for i in range(1000):
            output = model(input)

            if i % 1 == 0:
                plt.imshow(output.cpu().squeeze(), cmap='gray')
                plt.show()
            input = output
        

if __name__ == '__main__':
    main()