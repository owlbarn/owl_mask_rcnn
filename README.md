# Mask R-CNN

This is an implementation of the Mask R-CNN network using OCaml's numerical library [Owl](https://github.com/owlbarn/owl). This network can be used to perform object detection, segmentation and classification. The implementation is based on [this paper](https://arxiv.org/abs/1703.06870) and ported from [this Keras implementation](https://github.com/matterport/Mask_RCNN).

## Prerequisites
- OCaml `>=4.06.0`
- CamlImages (`opam install camlimages`). Note that you need to install it after installing the following packages `libpng12-dev libjpeg-dev libtiff-dev libxpm-dev libfreetype6-dev libgif-dev` to make it support the image format you are interested in.
- [Owl's **master branch**](https://github.com/owlbarn/owl) (make sure it is up-to-date)
- You need pre-trained weights to run the inference mode of the network. You can directly download the Owl weights **[here](https://drive.google.com/open?id=1MTnjFeSNB3Xuh471Lnk0iK-90AzTXf8k)** and place them at the root of the directory (they are converted from the Keras weights that can be found [here](https://github.com/matterport/Mask_RCNN/releases)).
- You can then `make` and `make run`!

## Images
![Image](https://github.com/pvdhove/owl-mask-rcnn/blob/master/results/buildings_1536.jpg)
The code `evalImage.ml` from the examples can be used to classify all the pictures in a given folder. It can be compiled with `make` and run with `make run`. A new image with highlighted objects will be generated to the `results/` folder. You can modify the location of the source directory/file in `examples/evalImage.ml`, as well as the size of the image: a larger size yields a more accurate detection but needs more time and memory (default is 768, but you can try 512, 1024, 1536, 2048,...).

## Videos
If you are patient enough, you can try to convert a video frame-by-frame by running `make video` (you need FFmpeg to run it). You can modify the location of the source video in `examples/evalVideo.ml`. Note that this writes all the frames of the video on the hard drive.
