# **MP-DistillFormer: Multimodal Prototype Distillation with Self-Supervised Transformer for Few-Shot Fine-Grained Classification**

## Environment
The code is tested on Windows 11 with Anaconda3 and following packages:
- python 3.11.5
- pytorch 2.4.0

## Preparation
1. Change the ROOT_PATH value in the following files to yours:
    - `datasets/miniimagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`
    - `datasets/fc100.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - ***CUB-200-2011**: download from [Caltech Vision Lab](https://www.vision.caltech.edu/datasets/cub_200_2011/) and put in `data/miniImageNet` folder.

    - ***Stanford Dogs**: download from [Stanford Artificial Intelligence](http://vision.stanford.edu/aditya86/ImageNetDogs/) and put in `data/tieredImageNet` folder.

    - **Stanford Cars**: download from [Stanford Education](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) and put in `data/cifarfs` folder.

  
   
Since the datasets are public, please download the raw images from their official websites:
- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
- Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
- Stanford Cars: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

To ensure exact reproducibility of our few-shot episodes, 
we have provided the data split files (`train.txt`, `valid.txt`, and `test.txt`) in the `./dataset/` directory.
 
## Pre-trained Models
[Optional] The pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1E1vj3dUH8Ilpn3LWNIQT1_hPxQkJDIR5). Extract and put the content in the `save` folder. To evaluate the model, run the `test.py` file with the proper save path as in the next section.

## Experiments:
To train on 5-way 1-shot and 5-way 5-shot Stanford Dogs:

```
python train_stage1.py --dataset dog --save-path ./save/stage1 --shot 1
python train_stage1.py --dataset dog --save-path ./save/stage1 --shot 5

python train_stage2.py --dataset dog --stage1-path ./stage1 --save-path ./save/stage2 --shot 1
python train_stage2.py --dataset dog --stage1-path ./stage1 --save-path ./save/stage2 --shot 5

python train_stage3.py --dataset dog --stage2-path ./stage2 --save-path ./save/stage3 --shot 1
python train_stage3.py --dataset dog --stage2-path ./stage2 --save-path ./save/stage3 --shot 5
```

To evaluate on 5-way 1-shot and 5-way 5-shot Stanford Dogs:
```
python test.py --dataset dog --shot 1 --save-path ./save/stage3
python test.py --dataset dog --shot 5 --save-path ./save/stage3
```
