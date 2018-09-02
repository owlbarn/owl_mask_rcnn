open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
module N = Dense.Ndarray.S

module RPN = RegionProposalNetwork
module PL = ProposalLayer
(*module PRA = PyramidROIAlign
module FPN = FeaturePyramidNetwork
module DL = DetectionLayer *)
module C = Configuration

(* Note: most function definitions only support batches of size 1, which is
 * enough for inference but not suitable for training. *)

(* *** MASK R-CNN *** *)
let mrcnn () =
  let () =
    if C.image_shape.(0) mod 64 <> 0 || C.image_shape.(1) mod 64 <> 0 then
      invalid_arg "Image width and height must be divisible by 64" in
  let nb_anchors = 261888 in
  (* compensates for the lack of Padding2D *)
  let input_shape = [|C.image_shape.(0) + 6; C.image_shape.(1) + 6; 3|] in
  let input_image = input ~name:"input_image" input_shape in

  (* The next two layers should be inputs of the network. Since Owl does not
   * support multi-inputs networks, we define them as input_image successors. *)
  let input_image_meta =
    lambda_array [|C.image_meta_size|]
      (fun t -> pack_arr (N.zeros [|(shape t.(0)).(0); C.image_meta_size|]))
      ~name:"input_image_meta" [|input_image|] in
  let anchors = lambda_array [|nb_anchors; 4|]
                  (fun t -> pack_arr (N.zeros [|(shape t.(0)).(0); nb_anchors; 4|]))
                  ~name:"input_anchors" [|input_image|] in

  let _, c2, c3, c4, c5 = Resnet.resnet101 input_image in

  let tdps = C.top_down_pyramid_size in
  let p5 = conv2d [|1; 1; 2048; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c5p5" c5 in
  let p4 =
    add ~name:"fpn_p4add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5;
        conv2d [|1; 1; 1024; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c4p4" c4|] in
  let p3 =
    add ~name:"fpn_p3add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4;
        conv2d [|1; 1; 512; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c3p3" c3|] in

  let p2 =
    add ~name:"fpn_p2add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3;
        conv2d [|1; 1; 256; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c2p2" c2|] in

  let p2 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p2" p2 in
  let p3 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p3" p3 in
  let p4 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p4" p4 in
  let p5 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p5" p5 in
  let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in

  let rpn_feature_maps = [|p2; p3; p4; p5; p6|] in
  let mrcnn_feature_maps = [|p2; p3; p4; p5|] in

  let nb_ratios = Array.length C.rpn_anchor_ratios in
  (* it should be possible to create this network once and use it 5 times *)
  (* removed rpn_class_logits because useless for inference *)
  let rpns = Array.init 5
               (fun i -> RPN.rpn_graph rpn_feature_maps.(i)
                           nb_ratios C.rpn_anchor_stride tdps
                           ("_p" ^ string_of_int (i + 2))) in
  let rpn_class = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(i).(0))) in
  let rpn_bbox = concatenate 1 ~name:"rpn_bbox"
                   (Array.init 5 (fun i -> rpns.(i).(1))) in

  let rpn_rois =
    let prop_f = PL.proposal_layer C.post_nms_rois C.rpn_nms_threshold in
    lambda_array [|C.post_nms_rois; 4|] prop_f ~name:"ROI"
      [|rpn_class; rpn_bbox; anchors|] in
(*
  let mrcnn_class_logits, mrcnn_class, mrcnn_bbox =
    FPN.fpn_classifier_graph rpn_rois mrcnn_feature_maps input_image_meta
      C.pool_size C.num_classes C.fpn_classif_fc_layers_size in

  let detection = lambda_array [|C.detection_max_instances; 6|]
                    (DL.detection_layer ()) ~name:"mrcnn_detection"
                    [|rpn_rois; mrcnn_class; mrcnn_bbox; input_image_meta|] in
  let detection_boxes = lambda_array [|C.detection_max_instances; 4|]
                          (fun x -> Maths.get_slice [[]; []; [0;4]] x.(0))
                          detection in

  let mrcnn_mask = FPN.build_fpn_mask_graph detection_boxes mrcnn_feature_maps
                     input_image_meta C.mask_pool_size C.num_classes in*)
  (* model? *)
  rpn_rois
