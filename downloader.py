import sys, tqdm
import subprocess

download_file = sys.argv[1]
dest_path = sys.argv[2]
file_read = open(download_file,'r').readlines()
ff1 = open('vid_error.txt','a')
ff2 = open('aud_error.txt','a')

for each_line in tqdm.tqdm(file_read):
    formatted_line = each_line[:-1]
    file_id = formatted_line.split('\t')[0]
    link = formatted_line.split('\t')[1]
    video_nm = dest_path + 'video/' + 'v_'+str(file_id)+'.mp4'
    audio_nm = dest_path + 'audio/' + 'a_'+str(file_id)+'.mp3'
    command1 = ['./youtube-dl', '--quiet', '--no-warnings', '-f', '"bestvideo[height<=240]"', 'mp4', '-o', '"%s"' % video_nm, '"%s"' % link]
    command1 = ' '.join(command1)

    command2 = ['./youtube-dl', '--quiet', '--no-warnings', '-f', '--extract-audio', '--audio-format', 'mp3', '-o', '"%s"' % audio_nm, '"%s"' % link]
    command2 = ' '.join(command2)

    print(command1)
    print(command2)
    break
    try:
        output = subprocess.check_output(command1, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        ff1.write(file_id+'\n')
    try:
        output = subprocess.check_output(command2, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        ff2.write(file_id+'\n')

ff1.close()
ff2.close()
    # print(command)
    # break
# print(download_file)
