(* Record of variables used to create the network *)
                           
let backbone_strides = [|4; 8; 16; 32; 64|]

let num_classes = 81

let rpn_anchor_stride = 1

let rpn_anchor_ratios = [|0.5; 1.; 2.|]

(* RGB *)
let mean_pixel = [|123.7; 116.8; 103.9|]

let image_dim = 1024

let image_shape = [|image_dim; image_dim; 3|]
                    
let image_meta_size = 1 + 3 + 3 + 4 + 1 + num_classes

let top_down_pyramid_size = 256

let post_nms_rois_inference = 1000

let rpn_nms_threshold = 0.7
                    
