# Tensorflow implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

<img src="" height="200px">

<img src="" height="200px">
<img src="" height="200px">

<img src="" height="200px">
<img src="" height="200px">

- input image size: 1024x768
- process time(CPU): 2.246 sec (Core i5-5257U)
- process time(GPU): 1.728 sec (GPU GRID K520)


## Requirement
- [Tensorflow](https://github.com/tensorflow/tensorflow)

```
$ pip install tensorflow
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolutional layers which are 10% of the entire model. 
The VGG model part in this implementation were based on [Tensorflow VGG16 and VGG19](https://github.com/machrisaa/tensorflow-vgg). 
PLEASE COMMENT the initialization function in `tensorflow-vgg/vgg16.py`, and also remember to download the npy file for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a>.

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download). 
Also, it will save the transformation model, including the trained weights, for later use (in C++) in ```graphs``` directory, while the checkpoint files would be saved in ```models``` directory. 
```
python train.py -s <style_image_path> -d <training_dataset_path> -g 0
```

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path>
```

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) hasn't been tested yet.

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Code structure written in this repository are based on following nice works, thanks to the author.

- [Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://github.com/yusuketomoto/chainer-fast-neuralstyle)
