open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

(* Should use the weight file only if the problem with the normalisation neuron
 * saving/loading is fixed. *)
(* let weight_file = "weights/mrcnn.network" *)
let src = "data/examples"
let out = "data/"

let rec process_dir f name =
  let is_dir =
    try Sys.is_directory name with
    | Sys_error e -> invalid_arg ("not a valid directory/file name: " ^ e) in
  if is_dir then
    Array.iter (fun d ->
        process_dir f (name ^ "/" ^ d)) (Sys.readdir name)
  else (
    Printf.printf "%s\n%!" name;
    f name
  )

let () =
  let fun_detect = Model.detect () in

  let eval src =
    let classes = MrcnnUtil.class_names in
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    if Array.length class_ids = 0 then
      Printf.printf "No objects detected on the picture :'(\n"
    else (
      let img_arr = Image.img_to_ndarray src in
      let filename = List.hd (List.rev (String.split_on_char '/' src)) in
      (* add the bounding boxes and the masks to the picture *)
      Visualise.display_masks img_arr rois masks;
      Image.save (out ^ filename) Images.Jpeg (Image.img_of_ndarray img_arr);
      (* display classes, confidence and position *)
      Array.iteri (fun i id ->
          Printf.printf "%13s: %.3f " classes.(id) (N.get scores [|i|]);
          let y1, x1, y2, x2 =
            N.(int_of_float rois.%{[|i; 0|]}, int_of_float rois.%{[|i; 1|]},
               int_of_float rois.%{[|i; 2|]}, int_of_float rois.%{[|i; 3|]}) in
          Printf.printf "at [(%4d, %4d), (%4d, %4d)]\n" y1 x1 y2 x2)
        class_ids
    )
  in
  process_dir eval src
