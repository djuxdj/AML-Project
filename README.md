# Multiclass Multilabel Untrimmed Video Event Detection (AML-Project)

* You can download the dataset using downloader.py, it takes command line parameter as txt files that conatins YouTube video links and destination directory to store the downloaded videos

* In order to extract audio features first download this file: [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)

* Then run gen_audioset_feature.py to obtain 128 dimensional embedding size for every 960ms

Before running preprocessing download this files
* [ResNet50 pretrained on ImageNet [PyTorch version]](https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth)
* [C3D pretrained on Sports1M [ported from Keras]](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)

* Run preprocess_frames.py present in preprocess/train and preprocess/test directory. Similarly you can run preprocess/{test/train}/create_output_super.py to obtain segment wise labels and preprocess/{test/train}/create_output_ws.py to obtain per video labels.

* For running the models you can go into the respective model directory and directly run model.py file.

* Similarly to run UCF-101 models, follow the same instructions as above. The model files are present in UCF-101-models directory

