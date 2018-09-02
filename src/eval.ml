module C = Configuration

open Owl
open Neural.S
module N = Dense.Ndarray.S

let weight_file = "weights/mrcnn.network"
let src = "pics/street.jpg"

(* Preprocessing recommended for Resnet. *)
let preprocess img =
  let img = N.copy img in
  let r = N.get_slice [[];[];[0]] img in
  let g = N.get_slice [[];[];[1]] img in
  let b = N.get_slice [[];[];[2]] img in

  let r = N.sub_scalar r 123.68 in
  let g = N.sub_scalar g 116.779 in
  let b = N.sub_scalar b 103.939 in

  N.set_slice [[];[];[0]] img b;
  N.set_slice [[];[];[1]] img g;
  N.set_slice [[];[];[2]] img r;
  img

let convert_to_ndarray src w h =
  let comp k n = (n lsr ((2 - k) lsl 3)) land 0x0000FF in (* get the kth color component *)
  let img = Images.load src [] in
  let img = match img with
    | Rgb24 map -> Rgb24.resize None map w h
    | _ -> invalid_arg "not implemented yet" in (* TODO *)
  let img_arr = Graphic_image.array_of_image (Rgb24 img) in
  N.init_nd [|w; h; 3|]
    (fun t -> float (comp t.(2) img_arr.(t.(0)).(t.(1))))

let () =
  let nn = Graph.get_network ~name:"Mask R-CNN" (Model.mrcnn ()) in
  Graph.init nn;
  (* Graph.load_weights nn weight_file; *)
  (* Graph.print nn; *)
  let img_size = 1024 in
  let img_arr = convert_to_ndarray src img_size img_size in
  (* quick hack to replace zero_padding2d *)
  let img_arr = N.pad ~v:0. [[3;3];[3;3]] img_arr in
  let img_arr = N.expand (preprocess img_arr) 4 in
  let result = Graph.model nn img_arr in
  N.print result
