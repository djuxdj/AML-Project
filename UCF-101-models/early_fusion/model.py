import torch, tqdm
import torch.nn as nn
from dataloader import Train_Dataset, Test_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='./logs/early_fusion')

class Activity_Detection(nn.Module):
    def __init__(self, input_size1, input_size2, projected_size, hidden_size, num_classes):
        super(Activity_Detection, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.projected_size = projected_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc11 = nn.Linear(input_size1, projected_size)
        self.fc22 = nn.Linear(input_size2, projected_size)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.lstm_f = nn.LSTMCell(projected_size*2, hidden_size)
        self.lstm_r = nn.LSTMCell(projected_size*2, hidden_size)
        self.lstm_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size*2, num_classes)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return Variable(d.data.new(bsz, self.hidden_size).zero_()), \
            Variable(d.data.new(bsz, self.hidden_size).zero_())

    def forward(self, resnet_features, c3d_features):
        out1 = self.fc11(resnet_features) 
        out1 = self.fc1_drop(out1)

        out2 = self.fc22(c3d_features)
        out2 = self.fc1_drop(out2)

        out3 = torch.cat([out1, out2], dim=2)

        fh, fc = self._init_lstm_state(out3)
        rh, rc = self._init_lstm_state(out3) 

        for i in range(15):
            fh, fc = self.lstm_f(out3[:,i,:], (fh, fc))
            fh = self.lstm_drop(fh)

        for i in reversed(range(15)):
            rh, rc = self.lstm_r(out3[:,i,:], (rh, rc))
            rh = self.lstm_drop(rh)

        out4 = torch.cat([fh,rh], dim=1)
        out4 = self.fc2(out4)

        return out4

def train():
    network = Activity_Detection(2048, 4096, 512, 128, 101).cuda()
    optimizer = optim.Adam(network.parameters(), lr = 3e-4)
    criterion=nn.CrossEntropyLoss()

    dataset_train = Train_Dataset(True)
    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=3)

    for epoch in range(100):
        epoch_loss = 0.0
        t_batch = 0.0
        network.train()
        for batch_index, data in tqdm.tqdm(enumerate(train_loader)):
            data_r, data_c, output = data
            data_r = data_r.cuda()
            data_r = torch.squeeze(data_r)
            data_c = data_c.cuda()
            data_c = torch.squeeze(data_c)
            output = output.cuda()
            output = torch.squeeze(output)

            optimizer.zero_grad()

            pred = network(data_r, data_c)
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
            data1_r, data1_c, output1 = data
            data1_r = data1_r.cuda()
            data1_c = data1_c.cuda()
            # data1 = torch.squeeze(data1)
            output1 = output1.cuda()
            # output1 = torch.squeeze(output1)
            pred = network(data1_r, data1_c)
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