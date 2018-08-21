module M = Model

open Owl
open Owl_types
open Neural 
open Neural.S
open Neural.S.Graph
       
let () =
  let nn = (M.resnet101 224 1000).(4) in
  print nn
