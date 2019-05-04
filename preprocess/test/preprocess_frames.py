
import os,tqdm
import cv2
import h5py
import numpy as np
import skimage
import torch
from torch.autograd import Variable
from model import AppearanceEncoder, MotionEncoder


dataset_path = '/home/pankaj/activity-net-videos/valid/video/'
file_data = os.listdir(dataset_path)
feature_size = 2048+4096
dest_path = '/home/pankaj/Activity-net-features/valid/video/'
max_frames = 20

def sample_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print('fps is '+str(fps))
    mid_frame = int(fps/2)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_iter = int(length/fps)

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        
    frames = np.array(frames)

    frame_list = []
    clip_list = [] 

    for i in range(num_iter):
        middle_index = mid_frame*(i+1)
        frame_list.append(frames[middle_index])
        clip_list.append(frames[middle_index-8: middle_index+8])

    frame_list = np.array(frame_list)
    clip_list = np.array(clip_list)
    indices = np.linspace(0, len(frame_list), max_frames, endpoint=False, dtype=int)
    final_frame_list = frame_list[indices]
    final_clip_list = clip_list[indices]
    return final_frame_list, final_clip_list

def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def extract_features(aencoder, mencoder):
    for each_line in tqdm.tqdm(file_data):
        print(each_line)
        vid_name = dataset_path+each_line
        frame_list, clip_list = sample_frames(vid_name)
        # print(frame_list.shape, clip_list.shape)
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        frame_list = frame_list.transpose((0, 3, 1, 2))
        frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()
        with torch.no_grad():
            af = aencoder(frame_list)

        clip_list = np.array([[resize_frame(x, 112, 112)
                               for x in clip] for clip in clip_list])
        clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
        clip_list = Variable(torch.from_numpy(clip_list), volatile=True).cuda()
        with torch.no_grad():
            mf = mencoder(clip_list)

        feats = torch.cat([af, mf], dim=1).data.cpu().numpy()
        file_nm = each_line[:-4]
        np.save(dest_path+file_nm+'.npy', feats)
        
def main():
    aencoder = AppearanceEncoder()
    aencoder.eval()
    aencoder.cuda()

    mencoder = MotionEncoder()
    mencoder.eval()
    mencoder.cuda()

    extract_features(aencoder, mencoder)

if __name__ == '__main__':
    main()