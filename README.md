nupic_imagenet
====

## The purpose of this repository
+ I'll verify the HTM using the task of ImageNet LSVRC-2014.

## Index
+ Dataset
+ Install nupic/pylearn2
+ Simple task

## Dataset
### CIFAR-10
+ The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
+ [The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)

### MNIST
+ The MNIST database of handwritten digits.
+ [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)

### ImageNet
+ What is ILSVRC2013
  + [Large Scale Visual Recognition Challenge 2014](http://www.image-net.org/challenges/LSVRC/2014/)

+  How to get images from imageNet
  + Get wnid list
    + You need to prepare csv file of wnid. 
    + I created the csv file by shaping LSVRC page.(http://image-net.org/challenges/LSVRC/2014/browse-synsets)
    + Here is sample csv file (data/classification_categorys.csv).

  + Run get_image_from_imagenet.py
    + This script to get the image from imagenet.
    + Simply, this script get image url by wnid, and download image file.
    + I have excluded the following file.
      + flicr not found image file
      + cannot open file
      + Too small file
      ```
      cd data
      python get_image_from_imagenet.py
      ```
      + [download-API](http://www.image-net.org/download-API)
        + wnids : http://image-net.org/challenges/LSVRC/2014/browse-synsets
        + image : http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=[wnid]
        + word  : http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=[wnid]


## Install nupic/pylearn2
### Install nupic/ 
+ [nupic](https://github.com/numenta/nupic)

### Install pylearn2
+ [install](http://deeplearning.net/software/pylearn2/index.html)
+ [tutorial](http://deeplearning.net/software/pylearn2/tutorial/)
  + When I execute make_dataset.py, the error has occurred. 
  + I edited train_example_path of pylearn2/scripts/tutorials/grbm_smd/make_dataset.py.
  ```
  IOError: permission error creating /Library/Python/2.7/site-packages/pylearn2/scripts/tutorials/grbm_smd/cifar10_preprocessed_train.pkl
  ```
  + environment values
  ```
  export PYLEARN2_DATA_PATH=/Users/karino-t/data
  export PYLEARN2_VIEWER_COMMAND="open -Wn"
  ```

## Technique
### [ImageNet Classification with Deep Convolutional Neural Networks]()
+ architecture
+ input
  + (3, 224, 224)
+ convolutional layer
  + 1. 96 kernel, (11, 11) max-pooling
  + 2. 256 kernel, (5,5,48) max-pooling 
  + 3. 384 kernel, (3,3,256)  all connect with second layer
  + 4. 384 kernel, (3,3,192)
  + 5. 384 kernel, (3,3,192) max-pooling 
+ full connected layer
  + 6. 2048+2048 sigmoid?, tanh?, other?
  + 7. 2048+2048
  + 8. softmax 1000
+ reducing overfiting
  + data augmentation
  + dropout (full connected layer ?)

### [Maxout Networks](http://jmlr.org/proceedings/papers/v28/goodfellow13.pdf)
+ 

### [Network in Network]()





## simple task
### CIFAR-10
+ datamodel  : cifar10.py
+ preprocess : make_dataset.py
+ model      : conv_sample.yaml
+   





