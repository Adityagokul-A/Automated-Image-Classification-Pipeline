import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import yaml

with open('config.yml','r') as conf:
    config_info = yaml.load(conf, Loader=yaml.SafeLoader)

DATASET_FOLDER = config_info['dataset_path']

#print(torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


transform = transforms.Compose(
    [
    transforms.Resize((500,500)),
    transforms.CenterCrop((360,360)),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)

BATCH_SIZE = config_info['batch_size']
NUM_WORKERS = config_info['num_workers']

dataset = datasets.ImageFolder(root=DATASET_FOLDER,transform=transform)
TRAINING_SET_SIZE = int(0.9 * len(dataset))

#maybe use sklearn train_Test_split
train_set, test_set = torch.utils.data.random_split(dataset, [TRAINING_SET_SIZE, len(dataset)-TRAINING_SET_SIZE])
train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=NUM_WORKERS)
classes = config_info['classes'] #sort this as directories are in sorted order
classes.sort()
#Show a sample batch
"""
def imshow(img):
    img = img*0.5 + 0.5 #de-Normalize
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(' '.join('%s' % classes[labels[j]] for j in range(BATCH_SIZE)))
"""

#Neural net definition

class ImageRecogNet(nn.Module):
    def __init__(self):
        super(ImageRecogNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3,8,4) 
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) 
        self.conv2 = nn.Conv2d(8,16,4)
        self.fc1 = nn.Linear(16 * 53 * 53, 4800)
        self.fc2 = nn.Linear(4800, 512)
        self.fc3 = nn.Linear(512,100)
        self.fc4 = nn.Linear(100, len(classes))

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = ImageRecogNet()
net = net.to(device)
print(net)

LEARNING_RATE = config_info['learning_rate']

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.8)
#optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def train():
    start.record()
    EPOCHS = config_info['epochs']
    for epoch in range(EPOCHS): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    end.record()
    torch.cuda.synchronize()

    # save
    MODEL_PATH = config_info['model_savepath']
    script_model = torch.jit.script(net)
    torch.jit.save(script_model,MODEL_PATH)


    print('Finished Training')
    print(start.elapsed_time(end))  # milliseconds




def test():
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    train()
    #test()