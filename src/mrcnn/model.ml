open Owl

module N = Dense.Ndarray.S

open CGraph
open Graph
open AD

module RPN = RegionProposalNetwork
module PL = ProposalLayer
module FPN = FeaturePyramidNetwork
module DL = DetectionLayer
module C = Configuration

(* Note: most function definitions only support batches of size 1, which is
 * enough for inference but not suitable for training. *)

(* *** MASK R-CNN *** *)

let mrcnn num_anchors =
  let image_shape = C.get_image_shape () in
  if image_shape.(0) mod 64 <> 0 || image_shape.(1) mod 64 <> 0 then
    invalid_arg "Image height and width must be divisible by 64";

  let inputs = inputs
                 ~names:[|"input_image"; "input_image_meta"; "input_anchors"|]
                 [|image_shape; [|C.image_meta_size|]; [|num_anchors; 4|]|] in
  let input_image = inputs.(0)
  and input_image_meta = inputs.(1)
  and input_anchors = inputs.(2) in

  (* ResNet: extracts features of the image (the first layers extract low-level
   * features, the last layers extract high-level features. *)
  let _, c2, c3, c4, c5 = Resnet.resnet101 input_image in

  (* Feature Pyramid Network: creates a second pyramid of feature maps from top
   * to bottom so that every map has access to high and low level features. *)
  let tdps = C.top_down_pyramid_size in
  let str = [|1; 1|] in
  let p5 = conv2d [|1; 1; 2048; tdps|] str ~padding:VALID ~name:"fpn_c5p5" c5 in
  let p4 =
    add ~name:"fpn_p4add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5;
        conv2d [|1; 1; 1024; tdps|] str ~padding:VALID ~name:"fpn_c4p4" c4|] in
  let p3 =
    add ~name:"fpn_p3add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4;
        conv2d [|1; 1; 512; tdps|] str ~padding:VALID ~name:"fpn_c3p3" c3|] in
  let p2 =
    add ~name:"fpn_p2add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3;
        conv2d [|1; 1; 256; tdps|] str ~padding:VALID ~name:"fpn_c2p2" c2|] in

  let conv_args = [|3; 3; tdps; tdps|] in
  let p2 = conv2d conv_args str ~padding:SAME ~name:"fpn_p2" p2 in
  let p3 = conv2d conv_args str ~padding:SAME ~name:"fpn_p3" p3 in
  let p4 = conv2d conv_args str ~padding:SAME ~name:"fpn_p4" p4 in
  let p5 = conv2d conv_args str ~padding:SAME ~name:"fpn_p5" p5 in
  let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in

  let rpn_feature_maps = [|p2; p3; p4; p5; p6|] in
  let mrcnn_feature_maps = [|p2; p3; p4; p5|] in

  (* Region Proposal Network: classifies each anchor as foreground or background
   * (not the exact class yet) and associate to each anchor a bounding box
   * refinement. *)
  let nb_ratios = Array.length C.rpn_anchor_ratios in
  (* removed rpn_class_logits because it is useless for inference *)
  let rpns = Array.init 5
               (fun i -> RPN.rpn_graph rpn_feature_maps.(i)
                           nb_ratios C.rpn_anchor_stride
                           ("_p" ^ string_of_int (i + 2))) in
  let rpn_class = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(i).(0))) in
  let rpn_bbox = concatenate 1 ~name:"rpn_bbox"
                   (Array.init 5 (fun i -> rpns.(i).(1))) in

  (* Proposal layer: selects the top anchors that don't overlap too much and
   * refines their bounding box. *)
  let rpn_rois =
    let prop_f = PL.proposal_layer C.post_nms_rois C.rpn_nms_threshold in
    MrcnnUtil.delay_lambda_array [|C.post_nms_rois; 4|] prop_f ~name:"ROI"
      [|rpn_class; rpn_bbox; input_anchors|] in

  (* Feature Pyramid Network Classifier: associates a class to each proposal
   * and refines the bounding box for that class even more. *)
  let mrcnn_class, mrcnn_bbox =
    FPN.fpn_classifier_graph rpn_rois mrcnn_feature_maps input_image_meta
      C.pool_size C.num_classes C.fpn_classif_fc_layers_size in

  let detections = MrcnnUtil.delay_lambda_array [|C.detection_max_instances; 6|]
                    (DL.detection_layer ()) ~name:"mrcnn_detection"
                    [|rpn_rois; mrcnn_class; mrcnn_bbox; input_image_meta|] in
  let detection_boxes = lambda_array [|C.detection_max_instances; 4|]
                          (fun t -> Maths.get_slice [[]; []; [0;3]] t.(0))
                          [|detections|] in

  (* Generates low resolution masks for each detected object. *)
  let mrcnn_mask = FPN.build_fpn_mask_graph detection_boxes mrcnn_feature_maps
                     input_image_meta C.mask_pool_size C.num_classes in

  outputs ~name:C.name [|detections; mrcnn_mask|]


(* *** Input and Output Processing *** *)

type results = {
    rois: N.arr;
    class_ids: int array;
    scores: N.arr;
    masks: int -> N.arr * int * int * int * int;
  }


(* Transforms the outputs of the network in an easily understandable format. *)
let extract_features detections mrcnn_masks image_meta =
  let meta = Image.parse_image_meta (N.expand image_meta 2) in
  let rois, class_ids, scores, masks =
    Image.unmold_detections
      (N.squeeze ~axis:[|0|] detections)
      (N.squeeze ~axis:[|0|] mrcnn_masks)
      meta.original_image_shape
      meta.image_shape
      meta.window in
  { rois; class_ids; scores; masks; }


let detect () =
  (* Generate the anchors. *)
  let anchors = N.expand (Image.get_anchors (C.get_image_shape ())) 3 in
  (* Build the network and load the weights. *)
  let nn = mrcnn (N.shape anchors).(1) in
  if not (Sys.file_exists C.weight_file) then
    failwith "You have to download the pre-trained weights here \
      https://drive.google.com/open?id=1MTnjFeSNB3Xuh471Lnk0iK-90AzTXf8k \
      and place them at the root of the owl-mask-rcnn directory.";
  Graph.load_weights nn C.weight_file;
  Owl_log.info "Weights loaded!";
  (* Optimise the network. *)
  let eval = CGraph.Compiler.model_inputs nn in
  Owl_log.info "Computation graph built!";

  (* Evaluate the inputs and returns the results. *)
  (fun src ->
    let molded_image, image_meta, _ = Image.mold_inputs src in
    let image = N.expand molded_image 4 in
    let image_meta = N.expand image_meta 2 in
    let inputs = Array.map MrcnnUtil.pack [|image; image_meta; anchors|] in
    let outputs = eval inputs |> Array.map MrcnnUtil.unpack in
    let results = extract_features outputs.(0) outputs.(1) image_meta in
    results
  )
