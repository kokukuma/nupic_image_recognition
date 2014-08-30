nupic_imagenet
====

## The purpose of this repository
+ I'll verify the HTM using the task of ImageNet LSVRC-2014.

## Index
+ imagenet
+ simple task

## ImageNet
### What is ILSVRC2013
+ [Large Scale Visual Recognition Challenge 2014](http://www.image-net.org/challenges/LSVRC/2014/)

### How to get images from imageNet
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

## simple task
+ 5 categorys


