open Owl
open Owl_types
open Neural 
open Neural.S
open Neural.S.Graph
module AD = Owl.Algodiff.S
module N = Dense.Ndarray.S
             
module C = Configuration

(* *** REGION PROPOSAL NETWORK *** 
 * Add different names for each p_i? *)
let rpn_graph feature_map anchors_per_location anchor_stride =
  let shared = conv2d [|3; 3; 256; 512|] [|anchor_stride; anchor_stride|] (* not 256 *)
                 ~padding:SAME ~act_typ:Activation.Relu ~name:"rpn_conv_shared"
                 feature_map in
  let x = conv2d [|1; 1; 512; 2 * anchors_per_location|] [|1; 1|]
            ~padding:VALID ~name:"rpn_class_raw" shared in
  (* Other function reshape I could use???? *)
  let rpn_class_logits = lambda (fun t -> AD.pack_arr
                                            (N.reshape (AD.unpack_arr t)
                                               [|(N.shape (AD.unpack_arr t)).(0); -1; 2|])) x in
  let rpn_probs = activation Activation.(Softmax 1)
                    ~name:"rpn_class_xxx" rpn_class_logits in
  let x = conv2d [|1; 1; 512; anchors_per_location * 4|] [|1; 1|] ~padding:VALID
            ~name:"rpn_bbox_pred" shared in
  let rpn_bbox = lambda (fun t -> AD.pack_arr
                                    (N.reshape (AD.unpack_arr t)
                                       [|(N.shape (AD.unpack_arr t)).(0); -1; 4|])) x in
  [|rpn_class_logits; rpn_probs; rpn_bbox|] (* rpn_class_logits might be useless for
                                             * inference *)

(* depth (and this function) might be useless *)
let build_rpn_model input_map anchor_stride anchors_per_location depth =
  let outputs = rpn_graph input_map anchors_per_location anchor_stride in
  outputs (* should be model input -> outputs... *)
    
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
  let rpns = Array.init 5
               (fun i -> build_rpn_model rpn_feature_maps.(i)
                           C.rpn_anchor_stride nb_ratios tdps) in
  let rpn_class = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(1).(i))) in
  let rpn_bbox = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(2).(i))) in
  let rpn_rois = (proposal_layer C.post_nms_rois_inference C.rpn_nms_threshold
                   ~name:"ROI") [|rpn_class; rpn_bbox; anchors|] in
  
  p2
  
