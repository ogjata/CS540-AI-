import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training:
        dataSet = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    else:
        dataSet = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=64)

    return dataLoader

def build_model():
    untrained_NN_model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    return untrained_NN_model

def train_model(model, train_loader, criterion, T):
    given_opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epochs in range(T):
        model.train()
        sample = 0
        correct_accuracy = 0
        total_running_loss = 0
    

        for data, target in train_loader:
            given_opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            given_opt.step()

            total_running_loss = total_running_loss + (loss.item() * data.size(0))
            irr, predicted = torch.max(output, 1)
            sample = sample + target.size(0)
            correct_accuracy = correct_accuracy + (predicted == target).sum().item()
 
        accper = 100.0 * correct_accuracy / sample
        avgloss = total_running_loss / sample

        print(f'Train Epoch: {epochs} Accuracy: {correct_accuracy}/{sample}({accper:.2f}%) Loss: {avgloss:.3f}')

    
def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    sample_size = 0
    num_correct = 0
    total_running_loss = 0

    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            loss = criterion(output, label).item()
            total_running_loss = total_running_loss + (loss * data.size(0))
            irr, prediction = torch.max(output, 1)
            sample_size = sample_size + label.size(0)
            num_correct = num_correct + (prediction == label).sum().item()

    accper = 100.0 * num_correct / sample_size

    if show_loss:
        avgloss = total_running_loss / sample_size
        print(f'Average loss: {avgloss:.4f}')
        print(f'Accuracy: {accper:.2f}%')
    else:
        print(f'Accuracy: {accper:.2f}%')

def predict_label(model, test_images, index):
    model.eval()

    with torch.no_grad():
        output_logits = model(test_images[index])
        given_prob = F.softmax(output_logits, dim=1)
        three_top_prob, three_top_index = torch.topk(given_prob, 3, dim=1)
        
        given_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        for i in range(3):
            class_index_label = three_top_index[0][i].item()
            given_class_prob_per = three_top_prob[0][i].item() * 100.0
            labelIndex_className = given_class_names[class_index_label]
            print(f'{labelIndex_className}: {given_class_prob_per:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
