open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

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
    Owl_log.info "Processing file: %s..." name;
    f name
  )

let () =
  let fun_detect = Model.detect () in

  let eval src =
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    if Array.length class_ids = 0 then
      Printf.printf "No objects detected on the picture :'(\n"
    else (
      let img_arr = Image.img_to_ndarray src in
      let filename = List.hd (List.rev (String.split_on_char '/' src)) in
      (* add the bounding boxes and the masks to the picture *)
      Visualise.display_masks img_arr rois masks class_ids;
      Image.save (out ^ filename) Images.Jpeg (Image.img_of_ndarray img_arr);
      (* display classes, confidence and position *)
      Visualise.print_results class_ids rois scores
    )
  in
  process_dir eval src
