module M = Model
module C = Configuration

open Owl
open Owl_types
open Neural 
open Neural.S
open Neural.S.Graph

let () =
  let nn = M.mrcnn () in
  Printf.printf "hey\n%!"; 
  print (get_network nn)
