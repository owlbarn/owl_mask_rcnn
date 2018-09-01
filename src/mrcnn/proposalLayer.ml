open Owl
module AD = Neural.S.Algodiff
module N = Dense.Ndarray.S

module C = Configuration

(* *** PROPOSAL LAYER *** *)
(* A box is givenc by [|y1; x1; y2; x2|] *)
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
    let inputs = Array.map AD.unpack_arr inputs in
    let scores = N.get_slice [[]; []; [1]] inputs.(0) in
    let deltas = N.(inputs.(1) * reshape C.rpn_bbox_std_dev [|1; 1; 4|]) in
    let anchors = inputs.(2) in

    let pre_nms_limit = min 6000 (N.shape anchors).(1) in
    let ix = N.top scores pre_nms_limit in
    let scores = N.init [|1; pre_nms_limit|] (fun i -> N.get scores (ix.(i))) in
    let deltas = N.init_nd [|pre_nms_limit; 4|]
                   (fun i -> N.get deltas [|1; ix.(i.(0)).(1); i.(1)|]) in
    let pre_nms_anchors = N.init_nd [|pre_nms_limit; 4|]
                            (fun i -> N.get anchors [|1; ix.(i.(0)).(1); i.(1)|]) in
    (* check that and factorise *)

    let boxes = apply_box_deltas_graph pre_nms_anchors deltas in
    let window = N.of_array [|0.; 0.; 1.; 1.|] [|4|] in
    let boxes = clip_boxes_graph boxes window in

    (* TODO: implement non maximum suppression to select the boxes !!! *)

    AD.pack_arr (N.expand boxes 3)
  )
