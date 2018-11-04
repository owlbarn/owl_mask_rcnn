open Owl

open CGraph.Graph
module N = Dense.Ndarray.Generic

open Hdf5_caml

let filename = "mask_rcnn_coco_owl.hdf5"
let out_name = Configuration.weight_file

let conv2d_W = "kernel:0"
let conv2d_b = "bias:0"
let bn_beta = "beta:0"
let bn_gamma = "gamma:0"
let bn_mu = "moving_mean:0"
let bn_var = "moving_variance:0"
let lin_W = "kernel:0"
let lin_b = "bias:0"

let load nn =
  let h5_file =
    try H5.open_rdonly filename with
    | Hdf5_raw.H5i.Fail
      -> invalid_arg
           "you should download the pre-trained Owl weights here \
            https://drive.google.com/file/d/1PMrPU-CQmW5dVlwNIPO4fbdW4AWdu02c/view \
            and place them at the root of the directory" in
  Array.iter (fun n ->
      let param = Neuron.save_weights n.neuron in
      let len = String.length n.name in
      let name = if String.sub n.name (len - 3) 2 = "_p"
                    && String.sub n.name 0 3 = "rpn" then
                   String.sub n.name 0 (len - 3)
                 else n.name in
      let neuron_name = Neuron.to_name n.neuron in
      if neuron_name = "conv2d" || neuron_name = "transpose_conv2d" ||
           neuron_name = "dilated_conv2d" then (
        let w = H5.read_float_genarray h5_file (name ^ conv2d_W) C_layout in
        let b = H5.read_float_genarray h5_file (name ^ conv2d_b) C_layout in
        let w = N.cast_d2s w and
            b = N.cast_d2s b in
        param.(0) <- MrcnnUtil.pack w; param.(1) <- MrcnnUtil.pack b;
        Neuron.load_weights n.neuron param
      )
      else if neuron_name = "normalisation" then (
        let b = H5.read_float_genarray h5_file (name ^ bn_beta) C_layout in
        let g = H5.read_float_genarray h5_file (name ^ bn_gamma) C_layout in
        let mu = H5.read_float_genarray h5_file (name ^ bn_mu) C_layout in
        let var = H5.read_float_genarray h5_file (name ^ bn_var) C_layout in
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
        param.(0) <- MrcnnUtil.pack b; param.(1) <- MrcnnUtil.pack g;
        param.(2) <- MrcnnUtil.pack mu; param.(3) <- MrcnnUtil.pack var;
        Neuron.load_weights n.neuron param;
      )
      else if neuron_name = "linear" then (
        let w = H5.read_float_genarray h5_file (name ^ lin_W) C_layout in
        let b = H5.read_float_genarray h5_file (name ^ lin_b) C_layout in
        let w = N.cast_d2s w and
            b = N.cast_d2s b in
        let b_dim = Array.append [|1|] (Dense.Ndarray.S.shape b) in
        let b = Dense.Ndarray.S.reshape b b_dim in
        param.(0) <- MrcnnUtil.pack w; param.(1) <- MrcnnUtil.pack b;
        Neuron.load_weights n.neuron param
      )
    ) nn.topo;
  save_weights nn out_name;
  H5.close h5_file
