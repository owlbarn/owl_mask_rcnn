open Owl
module N = Dense.Ndarray.S

(* Record of variables used to create the network *)

let name = "Mask R-CNN"

let image_dim = 512

(* Cannot be changed at the moment. *)
let batch_size = 1

(* Strides of each layer of the FPN pyramid of Resnet101. *)
let backbone_strides = [|4; 8; 16; 32; 64|]

let fpn_classif_fc_layers_size = 1024

let top_down_pyramid_size = 256

let num_classes = 81

let rpn_anchor_scales = [|32.; 64.; 128.; 256.; 512.|]

let rpn_anchor_ratios = [|0.5; 1.; 2.|]

let rpn_anchor_stride = 1

let rpn_nms_threshold = 0.7

let post_nms_rois = 1000

(* RGB *)
let mean_pixel = [|123.7; 116.8; 103.9|]

let pool_size = 7
let mask_pool_size = 14

let rpn_bbox_std_dev = N.of_array [|0.1; 0.1; 0.2; 0.2|] [|4|]
let bbox_std_dev = N.of_array [|0.1; 0.1; 0.2; 0.2|] [|4|]

let detection_max_instances = 100

(* You can decrease the detection confidence if you want to detect more
 * objects *)
let detection_min_confidence = 0.6

let detection_nms_threshold = 0.3

let image_shape = [|image_dim; image_dim; 3|]

let image_meta_size = 1 + 3 + 3 + 4 + 1 + num_classes
