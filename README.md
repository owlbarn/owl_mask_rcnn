# Mask R-CNN

This is an implementation of the Mask R-CNN network using OCaml's numerical library [Owl](https://github.com/owlbarn/owl). This network can be used to perform object detection, segmentation and classification. The implementation is based on [this paper](https://arxiv.org/abs/1703.06870) and ported from [this Keras implementation](https://github.com/matterport/Mask_RCNN).

## Instructions
You need CamlImages and [Owl's **master branch**](https://github.com/owlbarn/owl).

You need pre-trained weights to run the inference mode of the network. You can directly download the Owl weights **[here](https://drive.google.com/open?id=1MTnjFeSNB3Xuh471Lnk0iK-90AzTXf8k)** (they are converted from the Keras weights that can be found [here](https://github.com/matterport/Mask_RCNN/releases)).

## Images
![Image](https://github.com/pvdhove/owl-mask-rcnn/blob/master/results/architecture-billboards-buildings-1095901x3072.jpg)
The code `evalImage.ml` from the examples can be used to classify all the pictures in a given folder. It can be compiled with `make` and run with `make run`. A new image with highlighted objects will be generated to the `/data` folder. You can modify the location of the source directory/file in `examples/evalImage.ml`. You can also consider changing the size of the image in `src/configuration.ml`: a bigger size gives a more accurate detection but needs more time and memory (default is 512, but you can try 1024, 1536 or 2048).

## Videos
If you are patient enough, you can try to convert a video frame-by-frame by running `make video` (you need FFmpeg to run it). You can modify the location of the source video in `examples/evalVideo.ml`. Note that this writes all the frames of the video on the hard drive.
