import numpy as np
import os
import torch
import torch.utils.data as data
import  tom_lib.utils as utils


class Train_Dataset(data.Dataset):
	def __init__(self, model_type):
		self.train_path = '/home/pankaj/AML-project/train/'
		self.class_codes = np.load('/home/pankaj/AML-project/class_list.npy')
		self.train_list = np.load('/home/pankaj/AML-project/train_class_list.npy')

	def __getitem__(self, index):
		features = []
		output = []
		i = 0
		for each_class in self.class_codes:
			vid_nm = self.train_list[i][index].split('.')[0]
			resnet_feature = np.load(self.train_path+each_class+'/'+vid_nm+'.npy')[:,:2048]
			features.append(resnet_feature)
			output.append(i)
			i += 1
		features = torch.from_numpy(np.array(features))
		output = torch.from_numpy(np.array(output))
		
		return features, output

	def __len__(self):
		return 72

class Test_Dataset(data.Dataset):
	def __init__(self, model_type):
		self.test_path = '/home/pankaj/AML-project/test/'
		self.features = np.load(self.test_path+'features.npy')
		self.output = np.load(self.test_path+'output.npy')

	def __getitem__(self, index):
		return self.features[index][:,:2048], self.output[index]

	def __len__(self):
		return len(self.output)

# train_data = Train_Dataset(True)
# data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
# next(iter(data_loader))
