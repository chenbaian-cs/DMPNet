# DMPNet

Pytorch implementation for DMPNet: Dynamic Message Propagation Network for RGB-D Salient Object Detection.
# Requirements
* Python 3.7 <br>
* Pytorch 1.8.0 <br>
* Torchvision 0.9.0 <br>
* Cuda 11.1

# Usage
This is the Pytorch implementation of DMPNet. It has been trained and tested on Linux (Ubuntu20 + Cuda 11.1 + Python 3.7 + Pytorch 1.8),
and it can also work on Windows. 

## To Train 
* Download the pre-trained ImageNet [backbone](https://pan.baidu.com/s/1DUeaVXsnIUgDsHJHITsG3g), password: bd0z<br>(resnet101/resnet50, densenet161, vgg16 and vgg_conv1, whereas the latter already exists in the folder), and put it in the 'pretrained' folder.
* Download the [training dataset](https://pan.baidu.com/s/18uGBTtfatIdDjaYSOSMRuw), password: uw24<br> and modify the 'train_root' and 'train_list' in the `main.py`.


* Start to train with
```sh
python main.py --mode=train --arch=resnet --network=resnet101 --train_root=xx/dataset/RGBDcollection --train_list=xx/dataset/RGBDcollection/train.lst 
```

## To Test 
* Download the [testing dataset](https://pan.baidu.com/s/18uGBTtfatIdDjaYSOSMRuw), password: uw24<br> and have it in the 'dataset/test/' folder. 
* Download the [already-trained DMPNet pytorch model](https://pan.baidu.com/s/1luvvwuWj5yaI4TXbbWLMgQ), password: kas7<br> and modify the 'model' to its saving path in the `main.py`.
* Start to test with
```sh
python main.py --mode=test --arch=resnet --network=resnet101 --model=xx/JLDCF_resnet101.pth --sal_mode=LFSD  --test_folder=test/LFSD  
```
