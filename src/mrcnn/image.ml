open Owl
module N = Dense.Ndarray.S

(* this should be changed if batch size > 1 *)
let norm_boxes_graph boxes shape =
  let h, w = shape.(0), shape.(1) in
  let scale = N.((of_array [|h;w;h;w|] [|4|]) -$ 1.) in
  let shift = N.of_array [|0.;0.;1.;1.|] [|4|] in
  N.((boxes - shift) / scale)

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
