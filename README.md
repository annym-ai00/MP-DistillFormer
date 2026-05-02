# **MP-DistillFormer: Multimodal Prototype Distillation with Self-Supervised Transformer for Few-Shot Fine-Grained Classification**

## Data Preparation
Since the datasets are public, please download the raw images from their official websites:
- CUB-200-2011: https://www.vision.caltech.edu/datasets/cub_200_2011/
- Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
- Stanford Cars: http://ai.stanford.edu/~jkrause/cars/car_dataset.html

To ensure exact reproducibility of our few-shot episodes, 
we have provided the data split files (`train.txt`, `valid.txt`, and `test.txt`) in the `./dataset/` directory.

## Dataset Structure:
dataset/
- CUB_200_2011/
	- train/
  	- valid/
	- test/
- Stanford_Dogs/
  	- train/
	- valid/
	- test/
- Stanford_Cars/
  	- train/
	- valid/
	- test/

## How to Run:
1. Train Stage 1:
> python train_stage1.py --dataset dog --save-path ./stage1

2. Train Stage 2:
> python train_stage2.py --dataset dog --stage1-path ./stage1 --save-path ./stage2

3. Train Stage 3:
> python train_stage3.py --dataset dog --stage2-path ./stage2 --save-path ./stage3

4. Test Final Model:
> python test.py --dataset dog --save-path ./stage3

Pretrained checkpoints are available at:
Google Drive: https://drive.google.com/drive/folders/1E1vj3dUH8Ilpn3LWNIQT1_hPxQkJDIR5
