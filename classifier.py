import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# Define the neural network architecture
class GVHDPredictor(nn.Module):
    def __init__(self):
        super(GVHDPredictor, self).__init__()
        self.fc1 = nn.Linear(23, 16) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def normalize(input: pd.DataFrame):
    for col in input.columns:
        total = 0
        for i in range(len(input)):
            total += float(input[col].iloc[i])
        for i in range(len(input)):
            input[col].iloc[i] = float(input[col].iloc[i]) / float(total)
    return input

def process_data(filename: str):
    df = pd.read_csv(filename, header=None)
    input = df.iloc[1:, :23]
    output = df.iloc[1:, 23:24]
    input = normalize(input)
    input = input.to_numpy(dtype='float32')
    output = output.to_numpy(dtype='float32')
    return (torch.tensor(input[:200]), torch.tensor(output[:200]), 
            torch.tensor(input[200:]), torch.tensor(output[200:]))

def train_model(model: GVHDPredictor, input, labels, criterion, optimizer):
    epoch = 0
    while True:
        optimizer.zero_grad()
        output = model.forward(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        if loss <= 0.1:
            pickle.dump(model, open("model.pkl", "wb"))
            break
        epoch += 1

def evaluate_model(model, input, labels, criterion):
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for i in range(len(input)):
            output = model.forward(input[i])
            loss = criterion(output, labels[i])
            prediction = torch.round(output)
            if prediction == 1 and labels[i] == 1:
                tp += 1
            elif prediction == 0 and labels[i] == 0:
                tn += 1
            elif prediction == 1 and labels[i] == 0:
                fp += 1
            else:
                fn += 1
            print("Prediction: " + str(torch.round(output)))
            print("Actual: " + str(labels[i]))
            print(loss.item())
    
    print("True Positives: " + str(tp))
    print("True Negatives: " + str(tn))
    print("False Positives: " + str(fp))
    print("False Negatives: " + str(fn))


if __name__ == "__main__":
    input_train, output_train, input_test, output_test = process_data('publicationdata.csv')
    model = GVHDPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_model(model, input_train, output_train, criterion, optimizer)
    evaluate_model(model, input_test, output_test, criterion)
    