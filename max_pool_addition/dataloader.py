import numpy as np
import os
import torch
import torch.utils.data as data
import  tom_lib.utils as utils


class Train_Dataset(data.Dataset):
	def __init__(self, model_type):
		self.video_path = '/home/pankaj/Activity-net-features/train/video/'
		self.audio_path = '/home/pankaj/Activity-net-features/train/audio/'
		self.result_path = '/home/pankaj/Activity-net-features/train/output_super/'
		self.train_files = os.listdir('/home/pankaj/Activity-net-features/train/video/')
		self.train_ids = [i[2:-4] for i in self.train_files]
		self.num_train = len(self.train_files)
		self.train_path = '/home/pankaj/Activity-net-features/train/'

	def __getitem__(self, index):
		video_file = self.train_path + 'video/v_' + self.train_ids[index] +'.npy'
		video_data = np.load(video_file)

		audio_file = self.train_path + 'audio/a_' + self.train_ids[index] +'.npy'
		audio_data = np.load(audio_file)

		result_file = self.train_path + 'output_super/r_' + self.train_ids[index] +'.npy'
		result_data = np.load(result_file)

		resnet_features = video_data[:,:2048]
		c3d_features = video_data[:,2048:]

		resnet_features = torch.from_numpy(np.array(resnet_features))
		c3d_features = torch.from_numpy(np.array(c3d_features))
		audio_features = torch.from_numpy(np.array(audio_data))
		result_features = torch.from_numpy(np.array(result_data))

		return resnet_features, c3d_features, audio_features, result_features

	def __len__(self):
		return self.num_train

class Test_Dataset(data.Dataset):
	def __init__(self, model_type):
		self.video_path = '/home/pankaj/Activity-net-features/valid/video/'
		self.audio_path = '/home/pankaj/Activity-net-features/valid/audio/'
		self.result_path = '/home/pankaj/Activity-net-features/valid/output_ws/'
		self.train_files = os.listdir('/home/pankaj/Activity-net-features/valid/video/')
		self.train_ids = [i[2:-4] for i in self.train_files]
		self.num_train = len(self.train_files)
		self.train_path = '/home/pankaj/Activity-net-features/valid/'

	def __getitem__(self, index):
		video_file = self.train_path + 'video/v_' + self.train_ids[index] +'.npy'
		video_data = np.load(video_file)

		audio_file = self.train_path + 'audio/a_' + self.train_ids[index] +'.npy'
		audio_data = np.load(audio_file)

		result_file = self.train_path + 'output_ws/r_' + self.train_ids[index] +'.npy'
		result_data = np.load(result_file)

		resnet_features = video_data[:,:2048]
		c3d_features = video_data[:,2048:]

		resnet_features = torch.from_numpy(np.array(resnet_features))
		c3d_features = torch.from_numpy(np.array(c3d_features))
		audio_features = torch.from_numpy(np.array(audio_data))
		result_features = torch.from_numpy(np.array(result_data))

		return resnet_features, c3d_features, audio_features, result_features

	def __len__(self):
		return self.num_train

# train_data = Train_Dataset(True)
# data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
# next(iter(data_loader))
