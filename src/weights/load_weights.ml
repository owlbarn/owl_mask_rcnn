open Owl
open Owl_types
open Bigarray
open Hdf5_caml

open Neural 
open Neural.S 
open Neural.S.Graph
module N = Dense.Ndarray.Generic
module AD = Owl.Algodiff.S

let fname = "mask_rcnn_coco_owl.hdf5"
let h5_file = H5.open_rdonly fname

let conv2d_W = "kernel:0"
let conv2d_b = "bias:0"
let bn_beta = "beta:0"
let bn_gamma = "gamma:0"
let bn_mu = "moving_mean:0"
let bn_var = "moving_variance:0"
let lin_W = "kernel:0"
let lin_b = "bias:0"

(* A helper function *)
let print_array a = 
  Printf.printf "[|";
  Array.iter (fun x -> Printf.printf "%i; " x) a ;
  Printf.printf "|]\n"

let () =
  let nn = Model.mrcnn () in
  Graph.init nn;
  let nodes = nn.topo in
  Array.iter (fun n ->
      let param = Neuron.mkpar n.neuron in
      if Neuron.to_name n.neuron = "conv2d" then (
        let w = H5.read_float_genarray h5_file (n.name ^ conv2d_W) C_layout in
        let b = H5.read_float_genarray h5_file (n.name ^ conv2d_b) C_layout in
        let w = N.cast_d2s w and
            b = N.cast_d2s b in
        param.(0) <- AD.pack_arr w; param.(1) <- AD.pack_arr b;
        Neuron.update n.neuron param )
      else if Neuron.to_name n.neuron = "normalisation" then (
        let b = H5.read_float_genarray h5_file (n.name ^ bn_beta) C_layout in
        let g = H5.read_float_genarray h5_file (n.name ^ bn_gamma) C_layout in
        let mu = H5.read_float_genarray h5_file (n.name ^ bn_mu) C_layout in
        let var = H5.read_float_genarray h5_file (n.name ^ bn_var) C_layout in
        let b = N.cast_d2s b and
            g = N.cast_d2s g and
            mu = N.cast_d2s mu and
            var = N.cast_d2s var in
        let len = Dense.Ndarray.S.shape b in 
        let b = Dense.Ndarray.S.reshape b [|1;1;1;len.(0)|] in
        let len = Dense.Ndarray.S.shape g in 
        let g = Dense.Ndarray.S.reshape g [|1;1;1;len.(0)|] in
        let len = Dense.Ndarray.S.shape mu in 
        let mu = Dense.Ndarray.S.reshape mu [|1;1;1;len.(0)|] in
        let len = Dense.Ndarray.S.shape var in 
        let var = Dense.Ndarray.S.reshape var [|1;1;1;len.(0)|] in
        param.(0) <- AD.pack_arr b; param.(1) <- AD.pack_arr g;
        Neuron.update n.neuron param;
        (function Neuron.Normalisation a -> (a.mu <- (AD.pack_arr mu))) n.neuron;
        (function Neuron.Normalisation a -> (a.var <- (AD.pack_arr var))) n.neuron;)
      else if Neuron.to_name n.neuron = "linear" then (
        let w = H5.read_float_genarray h5_file (n.name ^ lin_W) C_layout in
        let b = H5.read_float_genarray h5_file (n.name ^ lin_b) C_layout in
        let w = N.cast_d2s w and
            b = N.cast_d2s b in
        let b_dim = Array.append [|1|] (Dense.Ndarray.S.shape b) in
        let b = Dense.Ndarray.S.reshape b b_dim in
        param.(0) <- AD.pack_arr w; param.(1) <- AD.pack_arr b;
        Neuron.update n.neuron param )
      else
        ()
    ) nodes;
  Graph.save nn "resnet.network";
  H5.close h5_file

