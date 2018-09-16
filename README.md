# Mask R-CNN

This is an implementation of the Mask R-CNN network using OCaml's numerical library Owl. This network can be used to perform object detection, segmentation and classification. The implementation is based on the paper https://arxiv.org/abs/1703.06870 and ported from the Keras implementation https://github.com/matterport/Mask_RCNN.

## Instructions
You need CamlImages, hdf5_caml (note that the 0.1.4 version does not work, you should probably use the 0.1.3 version) and obviously Owl.

You need pre-trained weights to run the inference mode of the network. You can either download the Keras weights [here](https://github.com/matterport/Mask_RCNN/releases) and convert them with `save_weights.py`, or directly download the Owl weights [here](https://drive.google.com/open?id=1PMrPU-CQmW5dVlwNIPO4fbdW4AWdu02c).
The code `evalImage.ml` from the examples can be used to classify all the pictures in a given folder. It can be compiled with `make` and run with `make run`. You can modify the name of the source directory in `evalImage.ml`. You can also consider changing the size of the image in `src/configuration.ml`: a bigger size gives a more accurate detection but needs more time and memory.
