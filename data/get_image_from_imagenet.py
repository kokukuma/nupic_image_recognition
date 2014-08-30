#!/usr/bin/python
# coding: utf-8

import os
import csv
import urllib2
import filecmp
from PIL import Image


#classification_categorys_csv = "classification_categorys.csv"
classification_categorys_csv = "tmp.csv"
download_dir                 = "./dataset/"
max_image_cnt_per_wnid       = 10
flicr_not_found              = 'flicr_not_found.jpg'


def main():
    """
    Download classification image from imagenet

    classification_categorys_csv was created by shaping the page
    http://image-net.org/challenges/LSVRC/2014/browse-synsets

    """
    def check_image_and_raise(image_file_name):
        # This image is flicr not fount image
        if filecmp.cmp(image_file_name, flicr_not_found):
            raise Exception, "This image is flicr not fount image"

        # This image is could't read
        Image.open(image_file_name)

        # The file size is too small
        if os.path.getsize(image_file_name) < 100:
            raise Exception, "The file size is too small"


    with open(classification_categorys_csv, "rb") as f:
        csvReader = csv.reader(f)
        csvReader.next()

        for row in csvReader:
            wnid   = row[0]
            labels = row[1]

            # create directory
            dldir  = download_dir + wnid + "/"
            if not os.path.isdir(dldir):
                os.mkdir(dldir)

            # get image urls
            dlurls = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=" + wnid
            response = urllib2.urlopen(dlurls)
            urls = [x.strip() for x in response.readlines() if not x.strip() == ""]

            # get image and save
            image_cnt = 0
            for i, url in enumerate(urls) :
                image_file_name = ""
                if image_cnt >= max_image_cnt_per_wnid:
                    break
                try:
                    print "%d / %d : %s" % (image_cnt, max_image_cnt_per_wnid, url)
                    res = urllib2.urlopen(url, timeout=3)
                    image_file_name = dldir + url.split('/')[-1]
                    with open(image_file_name, 'wb') as image_f:
                        image_f.write(res.read())
                    check_image_and_raise(image_file_name)
                    image_cnt += 1

                except Exception as e:
                    print '### failed :' + str(e)
                    if os.path.exists(image_file_name):
                        os.remove(image_file_name)
                    continue


if __name__ == "__main__":
    main()
