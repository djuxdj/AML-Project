import os,tqdm,json
import numpy as np

dest_path = '/home/pankaj/Activity-net-features/valid/output_ws/'

f = open('class.txt','r').readlines()
ff = json.loads(open('activity_net.v1-2.min.json').read())

video_files = os.listdir('/home/pankaj/Activity-net-features/valid/video') 
final_train_id = [i[2:-4] for i in video_files]
# print(final_train_id[0])
# print(f[0][:-1])
final_list = [i[:-1] for i in f]

for key, value in tqdm.tqdm(ff['database'].items()):
    file_mode = value['subset']
    embedd = np.zeros((100))
    if file_mode == 'validation':
        if key in final_train_id:
            label_list = value['annotations']
            for each_label in label_list:
                label_nm = each_label['label']
                index_id = final_list.index(label_nm)
                embedd[index_id] = 1
            np.save(dest_path+'r_'+key+'.npy',embedd)
            # break

