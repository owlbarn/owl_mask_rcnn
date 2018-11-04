open Owl

open CGraph.Neural
open CGraph.Graph

(* *** REGION PROPOSAL NETWORK ***
 * For each anchor (= bounding box) on the image, returns the likelihood that
 * it contains an object and a refinement for the anchor. *)

(* feature_map: backbone features [batch_size, H, W, D],
 * anchors_per_loc: number of anchors per pixel,
 * anchor_stride: spacing between anchors,
 * name: name of that feature_map.
 * Returns:
 * rpn_probs: [batch_size, H * W * anchors_per_loc, 2]
 * rpn_bbox: [batch_size, H * W * anchors_per_loc, (dx, dy, log(dh), log(dy)]
 *           Bounding box refinements. *)
let rpn_graph feature_map anchors_per_loc anchor_stride name =
  let h, w, d =
    let shape = Neuron.get_out_shape feature_map.neuron in
    shape.(0), shape.(1), shape.(2) in
  let shared = conv2d [|3; 3; d; 512|] [|anchor_stride; anchor_stride|]
                 ~padding:SAME ~act_typ:Activation.Relu
                 ~name:("rpn_conv_shared" ^ name) feature_map in
  let x = conv2d [|1; 1; 512; 2 * anchors_per_loc|] [|1; 1|]
            ~padding:VALID ~name:("rpn_class_raw" ^ name) shared in

  let rpn_class_logits = reshape [|anchors_per_loc * h * w; 2|] x in
  let rpn_probs = activation Activation.(Softmax 2)
                    ~name:("rpn_class_xxx" ^ name) rpn_class_logits in

  let x = conv2d [|1; 1; 512; anchors_per_loc * 4|] [|1; 1|] ~padding:VALID
            ~name:("rpn_bbox_pred" ^ name) shared in
  let rpn_bbox = reshape [|anchors_per_loc * h * w; 4|] x in
  (* rpn_class_logits is useless for inference *)
  [|(*rpn_class_logits;*) rpn_probs; rpn_bbox|]

