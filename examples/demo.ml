open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration


(* Script used for the web demo of Mask R-CNN. *)
let () = C.set_image_size 1024


let () =
  Owl_log.set_level FATAL;
  (* Build the network once. *)
  let fun_detect = Model.detect () in

  let eval src =
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    let img_arr = Image.img_to_ndarray src in
    let filename = Filename.basename src in
    (* add the bounding boxes and the masks to the picture *)
    Visualise.display_masks img_arr rois masks class_ids;
    let out_loc = "../frontend/results/" ^ filename in
    Image.save out_loc Images.Jpeg (Image.img_of_ndarray img_arr);
    if Array.length class_ids > 0 then
      (* display classes, confidence and position *)
      Visualise.print_results class_ids rois scores
    else Printf.printf "No objects detected.\n"
  in

  eval Sys.argv.(1)
