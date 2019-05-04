# AML-Project

You can download the dataset using downloader.py, it takes command line parameter as txt files that conatins YouTube video links and destination directory to store the downloaded videos

In order to extract audio features first download this file: [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)

Then run gen_audioset_feature.py to obtain 128 dimensional embedding size for every 960ms

Run preprocess_frames.py present in preprocess/train and preprocess/test directory. Similarly you can run preprocess/{test/train}/create_output_super.py to obtain segment wise labels and preprocess/{test/train}/create_output_ws.py to obtain per video labels.

For running the models you can go into the respective model directory and directly run model.py file.

