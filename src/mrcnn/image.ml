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

let parse_image_meta meta =
  { image_id             = N.get_slice [[];[0]]     meta;
    original_image_shape = N.get_slice [[];[1;3]]   meta;
    image_shape          = N.get_slice [[];[4;6]]   meta;
    window               = N.get_slice [[];[7;10]]  meta;
    scale                = N.get_slice [[];[11]]    meta;
    active_class_ids     = N.get_slice [[];[12;-1]] meta;
  }


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
  concatenate ~axis:1 [|y1; x1; y2; x2|]


(* Clip boxes to image boundaries.
 * Boxes: [N, [y1, x1, y2, x2]], window: [y1, x1, y2, x2]. *)
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
let norm_boxes boxes shape =
  let h, w = shape.(0), shape.(1) in
  let scale = N.((of_array [|h;w;h;w|] [|4|]) -$ 1.) in
  let shift = N.of_array [|0.;0.;1.;1.|] [|4|] in
  N.((boxes - shift) / scale)


let area box = (box.(2) -. box.(0)) *. (box.(3) -. box.(1))


(* Check that the point (y1, x1) is always the top left corner? *)
let intersection_over_union box1 box2 =
  let y1 = max box1.(0) box2.(0)
  and x1 = max box1.(1) box2.(1)
  and y2 = min box1.(2) box2.(2)
  and x2 = min box1.(3) box2.(3) in
  let area1, area2 = area box1, area box2 in
  let inter_area = max 0. (y2 -. y1) *. max 0. (x2 -. x1) in
  inter_area /. (area1 +. area2 -. inter_area +. 1e-6)


(* Greedily returns the indices of the boxes with the highest score that don't
 * have an intersection over union greater than iou_threshold with each
 * other. Takes at most max_output_size boxes. Assumes that the boxes are
 * ordered by score.
 * TODO is it possible to use vectorised ops to speed up the computation? *)
let non_max_suppression boxes max_output_size iou_threshold =
  let n = (N.shape boxes).(0) in
  let boxes = Array.init n (fun i ->
                  Array.init 4 (fun j -> N.get boxes [|i; j|])) in
  let selected = ref []
  and size = ref 0
  and i = ref 0 in
  while !i < n && !size < max_output_size do
    let ok = ref true
    and j = ref (!size - 1) in
    while !ok && !j >= 0 do
      if intersection_over_union boxes.(!i) boxes.(!j) > iou_threshold then
        ok := false;
      j := !j - 1;
    done;
    if !ok then (
      size := !size + 1;
      selected := !i :: !selected;
    );
    i := !i + 1;
  done;
  Array.of_list (List.rev !selected)
