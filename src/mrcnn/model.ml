open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
module N = Dense.Ndarray.S

module C = Configuration

let parse_image_meta_graph image =
  ()

(* *** PROPOSAL LAYER *** *)
(* A box has given by [|y1; x1; y2; x2|] *)
let apply_box_deltas_graph boxes deltas =
  let height = N.(get_slice [[]; [2]] boxes - get_slice [[]; [0]] boxes) in
  let width = N.(get_slice [[]; [3]] boxes - get_slice [[]; [1]] boxes) in
  let center_y = N.(get_slice [[]; [0]] boxes + (height *$ 0.5)) in
  let center_x = N.(get_slice [[]; [1]] boxes + (width *$ 0.5)) in

  let center_y = N.(center_y + ((get_slice [[]; [0]] deltas) * height)) in
  let center_x = N.(center_x + ((get_slice [[]; [1]] deltas) * width)) in
  let height = N.(height * exp (get_slice [[]; [2]] deltas)) in
  let width = N.(width * exp (get_slice [[]; [3]] deltas)) in

  let result = N.empty [|(N.shape boxes).(0); 4|] in
  N.(set_slice [[]; [0]] result (center_y - (height *$ 0.5)));
  N.(set_slice [[]; [1]] result (center_x - (width *$ 0.5)));
  N.(set_slice [[]; [2]] result (center_y + (height *$ 0.5)));
  N.(set_slice [[]; [3]] result (center_x + (width *$ 0.5)));
  result

let clip_boxes_graph boxes window =
  let edges = N.split [|1; 1; 1; 1|] window in
  let cols = N.split ~axis:1 [|1; 1; 1; 1|] boxes in
  (* relies on the broadcast operation *)
  let y1 = N.max2 (N.min2 cols.(0) edges.(2)) edges.(0) in
  let x1 = N.max2 (N.min2 cols.(1) edges.(3)) edges.(1) in
  let y2 = N.max2 (N.min2 cols.(2) edges.(2)) edges.(0) in
  let x2 = N.max2 (N.min2 cols.(3) edges.(3)) edges.(1) in

  let result = N.empty [|(N.shape boxes).(0); 4|] in
  N.set_slice [[]; [0]] result y1;
  N.set_slice [[]; [1]] result x1;
  N.set_slice [[]; [2]] result y2;
  N.set_slice [[]; [3]] result x2;
  result

let proposal_layer proposal_count nms_threshold =
  (fun inputs ->
    let scores = N.get_slice [[]; []; [1]] inputs.(0) in
    let deltas = N.(inputs.(1) * reshape C.rpn_bbox_std_dev [|1; 1; 4|]) in
    let anchors = inputs.(2) in
    
    let pre_nms_limit = min 6000 (N.shape anchors).(1) in
    let ix = N.top scores pre_nms_limit in
    let scores = N.init [|1; pre_nms_limit|] (fun i -> N.get scores (ix.(i))) in
    let deltas = N.init_nd [|1; pre_nms_limit; 4|]
                   (fun i -> N.get deltas [|1; ix.(i.(1)).(1); i.(2)|]) in
    let pre_nms_anchors = N.init_nd [|1; pre_nms_limit; 4|]
                            (fun i -> N.get anchors [|1; ix.(i.(1)).(1); i.(2)|]) in
    (* check that and factorise *)

    let boxes = N.empty [|1; pre_nms_limit; 4|] in
    N.iteri_slice ~axis:1
      (fun i t -> let box = apply_box_deltas_graph t (N.get_slice [[];[i];[]] deltas) in
                  N.set_slice [[];[i];[]] boxes box) pre_nms_anchors;
    let window = N.of_array [|0.; 0.; 1.; 1.|] [|4|] in
    N.iteri_slice ~axis:1
      (fun i t -> N.set_slice [[];[i];[]] boxes (clip_boxes_graph t window)) boxes;

    (* TODO: implement non maximum suppression to select the boxes !!! *)
    
    boxes
  )


(* *** ROIAlign Layer *** *)

let pyramid_roi_align pool_shape =
  (fun inputs ->
    let boxes = inputs.(0) in
    let image_meta = inputs.(1) in
    let feature_maps = Array.sub inputs 2 4 in

    let boxes = N.split ~axis:2 [|4|] boxes in (* check that *)
    let y1, x1, y2, x2 = boxes.(0), boxes.(1), boxes.(2), boxes.(3) in
    let h = N.(y2 - y1)
    and w = N.(x2 - x1) in
    let image_shape = parse_image_meta_graph(image_meta).image_shape.(0) in
    let image_area = image_shape.(0) *. image_shape.(1) in
    let roi_level = N.(log2 (sqrt (h * w) /$ (224. /. image_area))) in
    let roi_level = N.(min2 5. (max2 2. (roi_level +$ 4.))) in
    let roi_level = N.squeeze ~axis:[|2|] roi_level in

    let zero = N.zeros [|1|] in
    let pooled = Array.make 4 zero in
    let box_to_level = Array.make 4 zero in
    for level = 2 to 5 do
      (); (* implement bilinear crop and resize. *)
    done;

    let pooled = N.concatenate ~axis:0 pooled in
    let box_to_level = N.concatenate ~axis:0 box_to_level in
    (* TODO *)
    pooled
  )

  
(* *** REGION PROPOSAL NETWORK *** 
 * Add different names for each p_i? *)
let rpn_graph feature_map anchors_per_location anchor_stride =
  let shared = conv2d [|3; 3; 256; 512|] [|anchor_stride; anchor_stride|] (* not 256 *)
                 ~padding:SAME ~act_typ:Activation.Relu ~name:"rpn_conv_shared"
                 feature_map in
  let x = conv2d [|1; 1; 512; 2 * anchors_per_location|] [|1; 1|]
            ~padding:VALID ~name:"rpn_class_raw" shared in
  let rpn_class_logits = reshape [|512; -1; 2|] x in (* check the 512 *)
  let rpn_probs = activation Activation.(Softmax 1)
                    ~name:"rpn_class_xxx" rpn_class_logits in
  let x = conv2d [|1; 1; 512; anchors_per_location * 4|] [|1; 1|] ~padding:VALID
            ~name:"rpn_bbox_pred" shared in
  let rpn_bbox = reshape [|512; -1; 4|] x in (* check that 512 *)
  [|rpn_class_logits; rpn_probs; rpn_bbox|] (* rpn_class_logits might be useless for
                                             * inference *)

(* depth (and this function) might be useless *)
let build_rpn_model input_map anchor_stride anchors_per_location depth =
  let outputs = rpn_graph input_map anchors_per_location anchor_stride in
  outputs (* should be model input -> outputs... *)


(* *** Feature Pyramid Network *** *)
(* TODO: need TimeDistributed and PyramidROIAlign *)
let fpn_classifier_graph rois feature_maps image_meta
      pool_size num_classes fc_layers_size =
  let x =
    lambda [|1000; pool_size; pool_size; 256|] (pyramid_roi_align [|pool_size; pool_size|])
      ([|rois; image_meta|] @ feature_maps) ~name:"roi_align_classifier"
    |> conv2d [|pool_size; pool_size; 256; fc_layers_size|] [|1; 1|]
         ~padding:VALID ~name:"mrcnn_class_conv1"
    |> normalisation ~name:"mrcnn_class_bn1"
    |> activation Activation.Relu
    |> conv2d [|1; 1; fc_layers_size; fc_layers_size|] [|1; 1|]
         ~padding:VALID ~name:"mrcnn_class_conv2"
    |> normalisation ~name:"mrcnn_class_bn2"
    |> activation Activation.Relu in

  let shared = reshape [|1000; 1024|] ~name:"pool_squeeze" x in (* squeeze dim 2 and 3?*)
  let mrcnn_class_logits = linear num_classes ~name:"mrcnn_class_logits" shared in
  let mrcnn_probs = activation Activation.(Softmax 1)
                      ~name:"mrcnn_class" mrcnn_class_logits in
  let x = linear (num_classes * 4) ~name:"mrcnn_bbox_fc" shared in

  let mrcnn_bbox = reshape [|1000; num_classes; 4|] ~name:"mrcnn_bbox" x in (*check the 1000*)
  [|mrcnn_class_logits; mrcnn_probs; mrcnn_bbox|]

let build_fpn_mask_graph rois feature_maps image_meta pool_size num_classes =
  lambda [|100; pool_size; pool_size; 256|]
    (pyramid_roi_align [|pool_size; pool_size|]) ~name:"roi_align_mask"
  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv1"
  |> normalisation ~name:"mrcnn_mask_bn1"
  |> activation Activation.Relu
                
  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv2"
  |> normalisation ~name:"mrcnn_mask_bn2"
  |> activation Activation.Relu

  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv3"
  |> normalisation ~name:"mrcnn_mask_bn3"
  |> activation Activation.Relu
                
  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv4"
  |> normalisation ~name:"mrcnn_mask_bn4"
  |> activation Activation.Relu
                
  |> transpose_conv2d [|2; 2; 256; 256|] [|2; 2|]
       ~act_typ:Activation.Relu ~name:"mrcnn_mask_deconv"
  |> conv2d [|1; 1; 256; num_classes|] [|1; 1|]
       ~act_typ:Activation.Sigmoid ~name:"mrcnn_mask"
    
(* *** MASK R-CNN *** *)
let mrcnn () =
  let input_image = input ~name:"input_image" C.image_shape in
  let input_image_meta = input ~name:"input_image_meta" [|C.image_meta_size|] in
  let anchors = input ~name:"input_anchors" [|256; 4|] in (* 256? How many anchors?*)
  let _, c2, c3, c4, c5 = Resnet.resnet101 input_image in
  
  let tdps = C.top_down_pyramid_size in
  let p5 = conv2d [|1; 1; 2048; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c5p5" c5 in
  (* change this after you have upsampling2d *)
  let p4 =
    add ~name:"fpn_p4add"
      [|p5; (* up_sampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5 *) 
        conv2d [|1; 1; 1024; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c4p4" c4|] in
  let p3 =
    add ~name:"fpn_p3add"
      [|p4; (* up_sampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4 *)
        conv2d [|1; 1; 512; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c3p3" c3|] in
  let p2 =
    add ~name:"fpn_p2add"
      [|p3; (* up_sampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3 *)
        conv2d [|1; 1; 256; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c2p2" c2|] in
  let p2 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p2" p2 in
  let p3 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p3" p3 in
  let p4 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p4" p4 in
  let p5 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p5" p5 in

  let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in

  let rpn_feature_maps = [|p2; p3; p4; p5; p6|] in
  let mrcnn_feature_maps = [|p2; p3; p4; p5|] in

  let nb_ratios = Array.length C.rpn_anchor_ratios in
  
  (* it should be possible to create this network only once and to reuse it 5 times,
   * but I can't create a network with multiple outputs in Owl? *)
  let rpns = Array.init 5
               (fun i -> build_rpn_model rpn_feature_maps.(i)
                           C.rpn_anchor_stride nb_ratios tdps) in
  let rpn_class = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(1).(i))) in
  let rpn_bbox = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(2).(i))) in
  (* let rpn_rois = (proposal_layer C.post_nms_rois_inference C.rpn_nms_threshold
                   ~name:"ROI") [|rpn_class; rpn_bbox; anchors|] in *)
  rpn_class, rpn_bbox
  
