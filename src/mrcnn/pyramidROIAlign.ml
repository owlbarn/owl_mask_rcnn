open Owl
module AD = Neural.S.Algodiff
module N = Dense.Ndarray.S

module C = Configuration

(* *** ROIAlign Layer *** *)

let pyramid_roi_align pool_shape =
  fun inputs ->
    let inputs = Array.map AD.unpack_arr inputs in
    let boxes = inputs.(0) in
    let image_meta = inputs.(1) in
    let feature_maps = Array.sub inputs 2 4 in

    let boxes = N.split ~axis:2 [|4|] boxes in (* check that *)
    let y1, x1, y2, x2 = boxes.(0), boxes.(1), boxes.(2), boxes.(3) in
    let h = N.(y2 - y1)
    and w = N.(x2 - x1) in
    let image_shape = (Image.parse_image_meta_graph image_meta).image_shape in
    (* shape of the first image of batch, they should all have the same size *)
    let image_area = N.get image_shape [|0;0|] *. N.get image_shape [|0;1|] in
    let roi_level = N.(log2 (sqrt (h * w) /$ (224. /. image_area))) in
    let five = N.of_array [|5.|] [|1|]
    and two = N.of_array [|2.|] [|1|] in
    let roi_level = N.(min2 five (max2 two (roi_level +$ 4.))) in
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
    AD.pack_arr pooled

