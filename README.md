nupic_imagenet
====

## The purpose of this repository
+ I'll verify the HTM using the task of ImageNet LSVRC-2014.

## Index
+ imagenet
+ sample task

## ImageNet
### What is ILSVRC2013
+ [Large Scale Visual Recognition Challenge 2014](http://www.image-net.org/challenges/LSVRC/2014/)

### How to get images from imageNet
+ Get wnid list
  + You need to prepare csv file of wnid. 
  + I created the csv file by shaping LSVRC page.(http://image-net.org/challenges/LSVRC/2014/browse-synsets)
  + Here is sanple csv file(data/classification_categorys.csv).

+ Run get_image_from_imagenet.py
  + This program to get the image from imagenet. This program is to save the image to get the url of the image to wnid each.
    ```
    cd data
    python get_image_from_imagenet.py
    ```
    + [download-API](http://www.image-net.org/download-API)
      + wnids : http://image-net.org/challenges/LSVRC/2014/browse-synsets
      + image : http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=[wnid]
      + word  : http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=[wnid]

## small task


