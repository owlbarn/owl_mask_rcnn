open Owl
module N = Dense.Ndarray.S

module C = Configuration

(* *** PROPOSAL LAYER *** *)

(* Should be changed to support batch size > 1. *)
let proposal_layer proposal_count nms_threshold = fun inputs ->
  let inputs = Array.map MrcnnUtil.unpack inputs in
  (* [batch, number of ROIs, 1] where 1 is the foreground class confidence. *)
  let scores = N.get_slice [[]; []; [1]] inputs.(0) in
  (* N.print scores; *)
  (* [batch, number of ROIs, 4] *)
  let deltas = N.(inputs.(1) * C.rpn_bbox_std_dev) in
  (* N.print deltas; *)
  let anchors = inputs.(2) in

  let pre_nms_limit = min 6000 (N.shape anchors).(1) in
  let ix = N.top scores pre_nms_limit in
  let top_scores = N.init [|pre_nms_limit|] (fun i -> N.get scores ix.(i)) in

  let boxes =
    let deltas = N.init_nd [|pre_nms_limit; 4|]
                     (fun i -> N.get deltas [|0; ix.(i.(0)).(1); i.(1)|]) in
    let pre_nms_anchors =
      N.init_nd [|pre_nms_limit; 4|]
        (fun i -> N.get anchors [|0; ix.(i.(0)).(1); i.(1)|]) in
    let pre_boxes = Image.apply_box_deltas pre_nms_anchors deltas in
    let window = N.of_array [|0.; 0.; 1.; 1.|] [|4|] in
    Image.clip_boxes pre_boxes window in

  let proposals =
    let indices = Image.non_max_suppression
                    boxes top_scores proposal_count nms_threshold in
    let proposals = MrcnnUtil.gather_slice ~axis:0 boxes indices in (* to check *)
    let padding = max (proposal_count - (N.shape proposals).(0)) 0 in
    N.pad [[0; padding]; [0; 0]] proposals in

  MrcnnUtil.pack (N.expand proposals 3)
