module C = Configuration

open Owl
open Neural.S.Graph

let () =
  let nn = Model.mrcnn () in
  print (get_network nn)
