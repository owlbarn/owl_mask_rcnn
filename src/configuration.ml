open Owl
module N = Dense.Ndarray.S

(* Record of variables used to create the network *)

let name = "Mask R-CNN"

(* Images are resized to a square with this side length. The construction of
 * the network depends on this value. Higher means more accurate detections but
 * needs more time and memory. Must be divisible by 64. *)
let image_dim = ref 768

let get_image_size () = !image_dim
let set_image_size size = image_dim := size

let get_image_shape () =
  let size = get_image_size () in
  [|size; size; 3|]

(* Cannot be changed at the moment. *)
let batch_size = 1

(* Strides of each layer of the FPN pyramid of ResNet101. *)
let backbone_strides = [|4; 8; 16; 32; 64|]

(* Size of the fully connected layers. *)
let fpn_classif_fc_layers_size = 1024

(* Size of the layers to build the feature pyramid. *)
let top_down_pyramid_size = 256

(* Number of classes of objects to recognise (including background). *)
let num_classes = 81

(* Length of the anchors sides. *)
let rpn_anchor_scales = [|32.; 64.; 128.; 256.; 512.|]

(* Ratios of anchors width/height. *)
let rpn_anchor_ratios = [|0.5; 1.; 2.|]

(* Spacing between the anchors in pixels. *)
let rpn_anchor_stride = 1

(* Value (between 0 and 1) to filter RPN proposals. Filters out boxes that have
 * an intersection over union greater than this value. *)
let rpn_nms_threshold = 0.7

(* Maximum number of region of interests (ROIs) kept after non-maximum
 * suppression. *)
let post_nms_rois = 1000

(* Mean colours of the Coco dataset (RGB). *)
let mean_pixel = [|123.7; 116.8; 103.9|]

(* Size of the pooled ROIs. *)
let pool_size = 7
let mask_pool_size = 14

(* Standard deviation of the bounding boxes refinement. *)
let rpn_bbox_std_dev = N.of_array [|0.1; 0.1; 0.2; 0.2|] [|4|]
let bbox_std_dev = N.of_array [|0.1; 0.1; 0.2; 0.2|] [|4|]

(* Maximal number of objects to detect. *)
let detection_max_instances = 100

(* Only keeps objects whose detection confidence is higher that this value.
 * You can decrease it if you want to detect more objects. *)
let detection_min_confidence = 0.75

(* Same as the previous field but used as a first skimming. *)
let detection_nms_threshold = 0.3

(* Length of the image_meta array used to store information about the size of
 * the image and the active classes. *)
let image_meta_size = 1 + 3 + 3 + 4 + 1 + num_classes

(* Name of the file storing the weights. *)
let weight_file = "mrcnn_coco.weights"
