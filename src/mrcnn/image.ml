open Owl
module N = Dense.Ndarray.S

type image_meta =
  { image_id             : Dense.Ndarray.S.arr;
    original_image_shape : Dense.Ndarray.S.arr;
    image_shape          : Dense.Ndarray.S.arr;
    window               : Dense.Ndarray.S.arr;
    scale                : Dense.Ndarray.S.arr;
    active_class_ids     : Dense.Ndarray.S.arr;
  }

let parse_image_meta_graph meta =
  { image_id             = N.get_slice [[];[0]]     meta;
    original_image_shape = N.get_slice [[];[1;3]]   meta;
    image_shape          = N.get_slice [[];[4;6]]   meta;
    window               = N.get_slice [[];[7;10]]  meta;
    scale                = N.get_slice [[];[11]]    meta;
    active_class_ids     = N.get_slice [[];[12;-1]] meta;
  }


let non_max_suppression boxes scores max_output_size iou_threshold
      ?(score_threshold=neg_infinity) =
  ()


(* Applies the deltas to the boxes.
 * Boxes: [N, [y1, x1, y2, x2]], deltas: [N, [dx, dy, log(dh), log(dy)]]. *)
let apply_box_deltas boxes deltas =
  let open N in
  (* Convert to [y, x, h, w] *)
  let height = get_slice [[]; [2]] boxes - get_slice [[]; [0]] boxes in
  let width = get_slice [[]; [3]] boxes - get_slice [[]; [1]] boxes in
  let center_y = get_slice [[]; [0]] boxes + (height *$ 0.5) in
  let center_x = get_slice [[]; [1]] boxes + (width *$ 0.5) in
  (* Apply deltas *)
  let height = height * exp (get_slice [[]; [2]] deltas) in
  let width = width * exp (get_slice [[]; [3]] deltas) in
  let center_y = center_y + ((get_slice [[]; [0]] deltas) * height) in
  let center_x = center_x + ((get_slice [[]; [1]] deltas) * width) in
  (* Convert back *)
  let half_height = height *$ 0.5
  and half_width = width *$ 0.5 in
  let y1 = center_y - half_height
  and x1 = center_x - half_width
  and y2 = center_y + half_height
  and x2 = center_x + half_width in
  N.concatenate ~axis:1 [|y1; x1; y2; x2|]


(* Clip boxes to image boundaries. *)
let clip_boxes boxes window =
  let open N in
  let edges = split [|1; 1; 1; 1|] window in
  let cols = split ~axis:1 [|1; 1; 1; 1|] boxes in
  (* relies on the broadcast operation *)
  let y1 = max2 (min2 cols.(0) edges.(2)) edges.(0) in
  let x1 = max2 (min2 cols.(1) edges.(3)) edges.(1) in
  let y2 = max2 (min2 cols.(2) edges.(2)) edges.(0) in
  let x2 = max2 (min2 cols.(3) edges.(3)) edges.(1) in
  concatenate ~axis:1 [|y1; x1; y2; x2|]


(* this should be changed if batch size > 1 *)
let norm_boxes_graph boxes shape =
  let h, w = shape.(0), shape.(1) in
  let scale = N.((of_array [|h;w;h;w|] [|4|]) -$ 1.) in
  let shift = N.of_array [|0.;0.;1.;1.|] [|4|] in
  N.((boxes - shift) / scale)
