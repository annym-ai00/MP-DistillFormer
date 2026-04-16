[ReadMe.txt](https://github.com/user-attachments/files/26774582/ReadMe.txt)
MP-DistillFormer: Multimodal Prototype Distillation with Self-Supervised Transformer for Few-Shot Fine-Grained Classification

# Dataset Structure:
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

# How to Run:
1. Train Stage 1:
> python train_stage1.py --dataset dog --save-path ./stage1

2. Train Stage 2:
> python train_stage2.py --dataset dog --stage1-path ./stage1 --save-path ./stage2

3. Train Stage 3:
> python train_stage3.py --dataset dog --stage2-path ./stage2 --save-path ./stage3

4. Test Final Model:
> python test.py --dataset dog --save-path ./stage3


