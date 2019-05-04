import os,json,tqdm,math
import numpy as np

train_files = os.listdir('/home/pankaj/Activity-net-features/train/video') 
train_ids = [i[2:-4] for i in train_files]

dest_path = '/home/pankaj/Activity-net-features/train/output_super/'

f = open('class.txt','r').readlines()
label_list = [i[:-1] for i in f]
ff = json.loads(open('activity_net.v1-2.min.json').read())

max_frames = 20
# print(train_ids)

for key, value in tqdm.tqdm(ff['database'].items()):
    # print(value)
    file_mode = value['subset']
    if file_mode == 'training':
        if key in train_ids:
            # print(value)
            total_duration = int(math.floor(value['duration']))
            embedd = np.zeros((total_duration,101))
            # print(total_duration)
            segments = value['annotations']
            for each_segment in segments:
                seg_label = each_segment['label']
                seg = each_segment['segment']
                seg_start = int(math.ceil(seg[0]))
                seg_end = int(math.floor(seg[1]))
                seg_end = min(seg_end, total_duration)
                label_index = label_list.index(seg_label)
                for j in range(seg_start, seg_end):
                    embedd[j,label_index]=1
            for j in range(len(embedd)):
                if 1 not in embedd[j]:
                    embedd[j,100]=1
            indices = np.linspace(0, embedd.shape[0], max_frames, endpoint=False, dtype=int)
            new_embedd = embedd[indices]
            np.save(dest_path+'r_'+key+'.npy',new_embedd)
            # print(embedd[10])
                # print(embedd[10])
                # print(label_index)
                # print(seg_start, seg_end)