# Mask R-CNN

This is an implementation of the Mask R-CNN network using OCaml's numerical library [Owl](https://github.com/owlbarn/owl). This network can be used to perform object detection, segmentation and classification. The implementation is based on the paper https://arxiv.org/abs/1703.06870 and ported from the Keras implementation https://github.com/matterport/Mask_RCNN.

## Instructions
You need CamlImages, and [Owl's master branch](https://github.com/owlbarn/owl).

You need pre-trained weights to run the inference mode of the network. You can directly download the Owl weights [here](https://drive.google.com/open?id=1oGHoDktUGhRtTPAnaCJusb5xW_RHHEP5). They are converted from the Keras weights that can be found  [here](https://github.com/matterport/Mask_RCNN/releases).
The code `evalImage.ml` from the examples can be used to classify all the pictures in a given folder. It can be compiled with `make` and run with `make run`. You can modify the name of the source directory in `evalImage.ml`. You can also consider changing the size of the image in `src/configuration.ml`: a bigger size gives a more accurate detection but needs more time and memory.
