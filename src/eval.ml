module C = Configuration

open Owl
open Neural.S
module N = Dense.Ndarray.S

let weight_file = "weights/mrcnn.network"
let src = "pics/street.jpg"

let () =
  let fun_detect = Model.detect () in
  let classes = MrcnnUtil.class_names in
  let Model.({rois; class_ids; scores; masks}) = fun_detect src in

  let img_arr = Image.img_to_ndarray src in
  Visualise.display_masks img_arr rois masks;
  Image.save "pics/dest.jpg" Images.Jpeg (Image.img_of_ndarray img_arr);
  Array.iteri (fun i id ->
      Printf.printf "%s: %.3f" classes.(class_ids.(id)) N.(scores.%{[|i|]})) class_ids;
