open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

(* Should use the weight file only if the problem with the normalisation neuron
 * saving/loading is fixed. *)
(* let weight_file = "weights/mrcnn.network" *)
let src = "data/pics"
let out = "data/res/"

let rec process_dir f name =
  let is_dir =
    try Sys.is_directory name with
    | Sys_error e -> invalid_arg ("not a valid directory/file name: " ^ e) in
  if is_dir then
    Array.iter (fun d ->
        process_dir f (name ^ "/" ^ d)) (Sys.readdir name)
  else
    Printf.printf "%s\n%!" name;
    f name

let () =
  let fun_detect = Model.detect () in

  let eval src =
    let classes = MrcnnUtil.class_names in
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    if Array.length class_ids = 0 then
      Printf.printf "No objects detected on the picture :'(\n"
    else
      let img_arr = Image.img_to_ndarray src in
      let filename = List.hd (List.rev (String.split_on_char '/' src)) in
      Visualise.display_masks img_arr rois masks;
      Image.save (out ^ filename) Images.Jpeg (Image.img_of_ndarray img_arr);
      Array.iteri (fun i id ->
          Printf.printf "%s: %.3f\n" classes.(id) (N.get scores [|i|]))
        class_ids
  in
  process_dir eval src
