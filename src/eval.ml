module C = Configuration

open Owl
open Neural.S
module N = Dense.Ndarray.S

let weight_file = "weights/mrcnn.network"
let src = "pics/street.jpg"

let () =
  let fun_detect = Model.detect () in

  let result = fun_detect src in
  N.print result
