module C = Configuration

open Owl
open Neural.S
module N = Dense.Ndarray.S

let weight_file = "weights/mrcnn.network"
let src = "pics/street.jpg"

let () =
  let img_size = 1024 in
  (* check that and add it to the rest of the code *)
  let img_arr, meta, window = Image.mold_inputs src in
  (* quick hack to replace zero_padding2d *)
  let img_arr = N.pad ~v:0. [[3;3];[3;3]] img_arr in
  let img_arr = N.expand img_arr 4 in

  let nn = Graph.get_network ~name:"Mask R-CNN" (Model.mrcnn ()) in
  Graph.init nn;
  (* Graph.load_weights nn weight_file; *)
  (* Graph.print nn; *)
  let result = Graph.model nn img_arr in
  N.print result
