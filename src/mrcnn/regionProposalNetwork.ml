open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff

(* *** REGION PROPOSAL NETWORK ***
 * Different names for each p_i? *)
let rpn_graph feature_map anchors_per_loc anchor_stride depth name =
  let shared = conv2d [|3; 3; depth; 512|] [|anchor_stride; anchor_stride|]
                 ~padding:SAME ~act_typ:Activation.Relu
                 ~name:("rpn_conv_shared"^name) feature_map in
  let x = conv2d [|1; 1; 512; 2 * anchors_per_loc|] [|1; 1|]
            ~padding:VALID ~name:("rpn_class_raw"^name) shared in
  let rpn_class_logits = reshape [|-1; 2|] x in
  let rpn_probs = activation Activation.(Softmax 1)
                    ~name:("rpn_class_xxx"^name) rpn_class_logits in
  let x = conv2d [|1; 1; 512; anchors_per_loc * 4|] [|1; 1|] ~padding:VALID
            ~name:("rpn_bbox_pred"^name) shared in
  let rpn_bbox = reshape [|-1; 4|] x in
  [|(*rpn_class_logits;*)rpn_probs; rpn_bbox|]
  (* rpn_class_logits is useless for inference *)

(* this function might be useless *)
let build_rpn_model input_map anchor_stride anchors_per_loc depth name =
  let outputs = rpn_graph input_map anchors_per_loc anchor_stride depth name in
  outputs
