open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

(* Script to display the results of the evaluation of Mask R-CNN on images.
 * You can modify the three following variable to suit your needs. *)

(* Your image will be resized to a square of this dimension before being fed
 * to the network. It has to be a multiple of 64. A larger size means a more
 * accurate result but more time and memory to process. *)
let () = C.set_image_size 1024


let () =
  (* Build the network once. *)
  Owl_log.set_level FATAL;
  let fun_detect = Model.detect () in

  let eval src =
    let Model.({rois; class_ids; scores; masks}) = fun_detect src in
    let img_arr = Image.img_to_ndarray src in
    let filename = Filename.basename src in
    (* add the bounding boxes and the masks to the picture *)
    Visualise.display_masks img_arr rois masks class_ids;
    let out_loc = "results/" ^ filename in
    Image.save out_loc Images.Jpeg (Image.img_of_ndarray img_arr);
    (* display classes, confidence and position *)
    Printf.printf "%s\n" out_loc;
    Visualise.results_to_json class_ids rois scores
  in

  eval Sys.argv.(1)
