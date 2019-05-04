import torch, tqdm
import torch.nn as nn
from dataloader import Train_Dataset, Test_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchnet import meter

writer = SummaryWriter(log_dir='/home/pankaj/AML-Models/Supervised/logs/max_pool_addition')

class Activity_Detection(nn.Module):
    def __init__(self, resnet_size, c3d_size, audio_size, projected_size, hidden_size, num_classes):
        super(Activity_Detection, self).__init__()
        self.hidden_size = hidden_size

        self.fc_dropout = nn.Dropout(p=0.3)
        self.fc_audio = nn.Linear(audio_size, projected_size)
        self.fc_resnet = nn.Linear(resnet_size, projected_size)
        self.fc_c3d = nn.Linear(c3d_size, projected_size)

        self.lstm_audio_fwd = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_audio_rev = nn.LSTMCell(projected_size, hidden_size)

        self.lstm_resnet_fwd = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_resnet_rev = nn.LSTMCell(projected_size, hidden_size)

        self.lstm_c3d_fwd = nn.LSTMCell(projected_size, hidden_size)
        self.lstm_c3d_rev = nn.LSTMCell(projected_size, hidden_size)

        self.lstm_drop = nn.Dropout(p=0.3)

        self.fc_out = nn.Linear(hidden_size*2, num_classes)

    def _init_lstm_state(self, d):
        bsz = d.size(0)
        return Variable(d.data.new(bsz, self.hidden_size).zero_()), \
            Variable(d.data.new(bsz, self.hidden_size).zero_())

    def forward(self, resnet_features, c3d_features, audio_features):
        out_audio = self.fc_audio(audio_features)
        out_audio = self.fc_dropout(out_audio)

        out_resnet = self.fc_resnet(resnet_features)
        out_resnet = self.fc_dropout(out_resnet)

        out_c3d = self.fc_c3d(c3d_features)
        out_c3d = self.fc_dropout(out_c3d)

        audio_h_fwd, audio_c_fwd = self._init_lstm_state(out_audio)
        audio_h_rev, audio_c_rev = self._init_lstm_state(out_audio)

        resnet_h_fwd, resnet_c_fwd = self._init_lstm_state(out_resnet)
        resnet_h_rev, resnet_c_rev = self._init_lstm_state(out_resnet)

        c3d_h_fwd, c3d_c_fwd = self._init_lstm_state(out_c3d)
        c3d_h_rev, c3d_c_rev = self._init_lstm_state(out_c3d)

        audio_forward = []
        audio_backward = []

        resnet_forward = []
        resnet_backward = []

        c3d_forward = []
        c3d_backward = []

        for i in range(20):
            audio_h_fwd, audio_c_fwd = self.lstm_audio_fwd(out_audio[:,i,:], (audio_h_fwd, audio_c_fwd))
            audio_h_fwd = self.lstm_drop(audio_h_fwd)
            audio_forward.append(torch.unsqueeze(audio_h_fwd, 1))

        for i in reversed(range(20)):
            audio_h_rev, audio_c_rev = self.lstm_audio_rev(out_audio[:,i,:], (audio_h_rev, audio_c_rev))
            audio_h_rev = self.lstm_drop(audio_h_rev)
            audio_backward.append(torch.unsqueeze(audio_c_rev, 1))

        for i in range(20):
            resnet_h_fwd, resnet_c_fwd = self.lstm_resnet_fwd(out_resnet[:,i,:], (resnet_h_fwd, resnet_c_fwd))
            resnet_h_fwd = self.lstm_drop(resnet_h_fwd)
            resnet_forward.append(torch.unsqueeze(resnet_h_fwd, 1))

        for i in reversed(range(20)):
            resnet_h_rev, resnet_c_rev = self.lstm_resnet_rev(out_resnet[:,i,:], (resnet_h_rev, resnet_c_rev))
            resnet_h_rev = self.lstm_drop(resnet_h_rev)
            resnet_backward.append(torch.unsqueeze(resnet_h_rev, 1))

        for i in range(20):
            c3d_h_fwd, c3d_c_fwd = self.lstm_c3d_fwd(out_c3d[:,i,:], (c3d_h_fwd, c3d_c_fwd))
            c3d_h_fwd = self.lstm_drop(c3d_h_fwd)
            c3d_forward.append(torch.unsqueeze(c3d_h_fwd, 1))

        for i in reversed(range(20)):
            c3d_h_rev, c3d_c_rev = self.lstm_c3d_rev(out_c3d[:,i,:], (c3d_h_rev, c3d_c_rev))
            c3d_h_rev = self.lstm_drop(c3d_h_rev)
            c3d_backward.append(torch.unsqueeze(c3d_h_rev, 1))
        # print(audio_forward.size())
        audio_forward_complete = torch.cat(audio_forward, dim=1)
        audio_backward_complete = torch.cat(audio_backward, dim=1)
        audio_complete = torch.cat([audio_forward_complete, audio_backward_complete], dim=2)
        # print(audio_complete.size())

        resnet_forward_complete = torch.cat(resnet_forward, dim=1)
        resnet_backward_complete = torch.cat(resnet_backward, dim=1)
        resnet_complete = torch.cat([resnet_forward_complete, resnet_backward_complete], dim=2)

        c3d_forward_complete = torch.cat(c3d_forward, dim=1)
        c3d_backward_complete = torch.cat(c3d_backward, dim=1)
        c3d_complete = torch.cat([c3d_forward_complete, c3d_backward_complete], dim=2)

        features_size = audio_complete.shape[1]

        predictions = []

        for i in range(features_size):
            fused_out = audio_complete[:,i,:] + resnet_complete[:,i,:] + c3d_complete[:,i,:]
            predictions.append(torch.unsqueeze(self.fc_out(fused_out), 1))

        final_predictions = torch.cat(predictions, dim=1)
        return final_predictions
    
def train():
    network = Activity_Detection(2048, 4096, 128, 300, 128, 101).cuda()
    optimizer = optim.Adam(network.parameters(), lr = 3e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset_train = Train_Dataset(True)
    train_loader = DataLoader(dataset_train, batch_size=100, shuffle=True, num_workers=3)

    for epoch in range(1000):
        epoch_loss = 0.0
        t_batch = 0.0
        network.train()
        for batch_index, data in tqdm.tqdm(enumerate(train_loader)):
            data_r, data_c, data_a, output = data
            data_r = data_r.cuda()
            data_c = data_c.cuda()
            data_a = data_a.cuda()
            output = output.type(torch.cuda.FloatTensor)
            final_output = output.view(-1, 101)
            optimizer.zero_grad()
            # print(data_r.size(), data_c.size(), data_a.size(), output.size())
            pred = network(data_r, data_c, data_a)
            final_pred = pred.view(-1, 101)
            loss = criterion(final_pred, final_output)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            t_batch += 1
        epoch_loss = epoch_loss/t_batch
        writer.add_scalar('Epoch Training Loss', epoch_loss, epoch)
        print(epoch_loss)

        network.eval()
        dataset_test = Test_Dataset(True)
        test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True, num_workers=3)

        mtr = meter.APMeter()
        for batch_index, data in tqdm.tqdm(enumerate(test_loader)):
            data1_r, data1_c, data1_a, output1 = data
            data1_r = data1_r.cuda()
            data1_c = data1_c.cuda()
            data1_a = data1_a.cuda()
            # data1 = torch.squeeze(data1)
            with torch.no_grad():
                output1 = output1.type(torch.cuda.FloatTensor)
                pred = network(data1_r, data1_c, data1_a)
                max_pooling = nn.MaxPool2d((20,1))
                pred = torch.squeeze(max_pooling(pred))
                sig_layer = nn.Sigmoid()
                pred = sig_layer(pred)
                pred = pred[:,:100]
                # print(pred)
                # print(pred.size())
                mtr.add(pred, output1)
        map_value = mtr.value().mean()
        writer.add_scalar('mAP', map_value, epoch)
        print(map_value)

train()