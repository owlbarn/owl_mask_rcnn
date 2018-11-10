open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

(* Script to display the results of the evaluation of Mask R-CNN on images.
 * You can modify the three following variable to suit your needs. *)

(* Your image will be resized to a square of this dimension before being fed
 * to the network. It has to be a multiple of 64. A larger size means a more
 * accurate result but more time and memory to process. *)
let () = C.set_image_size 768

(* The location of the folder containing the pictures to process, or just the
 * filename of a picture. *)
let src = "data/examples"

(* The location of the folder to store the results. *)
let out = "results/"


(* [process_dir f dir] applies [f] on all the elements of a directory
 * recursively. *)
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
  (* Build the network once. *)
  let fun_detect = Model.detect () in

  let eval src =
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    if Array.length class_ids = 0 then
      Printf.printf "No objects detected on the picture.\n"
    else (
      let img_arr = Image.img_to_ndarray src in
      let filename = Filename.basename src |> Filename.remove_extension in
      (* add the bounding boxes and the masks to the picture *)
      Visualise.display_masks img_arr rois masks class_ids;
      let out_loc = out ^ filename ^ ".jpg" in
      Image.save out_loc Images.Jpeg (Image.img_of_ndarray img_arr);
      Owl_log.info "Output picture written to %s." out_loc;
      (* display classes, confidence and position *)
      Visualise.print_results class_ids rois scores
    )
  in

  process_dir eval src
