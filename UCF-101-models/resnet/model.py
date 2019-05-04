import torch, tqdm
import torch.nn as nn
from dataloader import Train_Dataset, Test_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='./logs/resnet')

class Activity_Detection(nn.Module):
    def __init__(self, input_size, projected_size, hidden_size, num_classes):
        super(Activity_Detection, self).__init__()
        self.input_size = input_size
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, projected_size)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.lstm_f = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_r = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size*2, num_classes)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return Variable(d.data.new(bsz, self.hidden_size).zero_()), \
            Variable(d.data.new(bsz, self.hidden_size).zero_())

    def forward(self, features):
        out1 = self.fc1(features) 
        out1 = self.fc1_drop(out1)

        fh, fc = self._init_lstm_state(out1)
        rh, rc = self._init_lstm_state(out1) 

        for i in range(15):
            fh, fc = self.lstm_f(out1[:,i,:], (fh, fc))
            fh = self.lstm_drop(fh)

        for i in reversed(range(15)):
            rh, rc = self.lstm_r(out1[:,i,:], (rh, rc))
            rh = self.lstm_drop(rh)

        out2 = torch.cat([fh,rh], dim=1)
        out2 = self.fc2(out2)

        return out2

def train():
    network = Activity_Detection(2048, 512, 128, 101).cuda()
    optimizer = optim.Adam(network.parameters(), lr = 3e-4)
    criterion=nn.CrossEntropyLoss()

    dataset_train = Train_Dataset(True)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=3)

    for epoch in range(100):
        epoch_loss = 0.0
        t_batch = 0.0
        network.train()
        for batch_index, data in tqdm.tqdm(enumerate(train_loader)):
            data, output = data
            data = data.cuda()
            data = torch.squeeze(data)
            output = output.cuda()
            output = torch.squeeze(output)

            optimizer.zero_grad()

            pred = network(data)
            loss = criterion(pred, output)

            epoch_loss += loss

            loss.backward()
            optimizer.step()

            t_batch += 1
            
        epoch_loss = epoch_loss/t_batch
        writer.add_scalar('Epoch Training Loss', epoch_loss, epoch)
        print(epoch_loss)

        network.eval()
        dataset_test = Test_Dataset(True)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=3)
        correct = 0.0
        total = 0.0
        for batch_index, data in tqdm.tqdm(enumerate(test_loader)):
            data1, output1 = data
            data1 = data1.cuda()
            # data1 = torch.squeeze(data1)
            output1 = output1.cuda()
            # output1 = torch.squeeze(output1)
            pred = network(data1)
            pred = nn.functional.softmax(pred, dim=1)

            pred_class = torch.argmax(pred, dim=1)

            pred_class = int(pred_class.item())
            actual_class = int(output1.item())
            
            if pred_class==actual_class:
                correct += 1
            total += 1

        accuracy = correct/total
        writer.add_scalar('Test Accuracy', accuracy, epoch)
        print(accuracy)

train()