open Owl

module N = Dense.Ndarray.S

open Mrcnn
module C = Configuration

let () = C.set_image_size 768

(* The location of the folder containing the pictures to process, or just the
 * filename of a picture. *)
let src = "data/test/street.jpg"


(* Code to measure performance copied from
 * https://github.com/owlbarn/benchmark/blob/master/core_ops/op_eval.ml *)
let remove_outlier arr =
  let first_perc = Owl_stats.percentile arr 25. in
  let third_perc = Owl_stats.percentile arr 75. in
  let lst = Array.to_list arr in
  List.filter (fun x -> (x >= first_perc) && (x <= third_perc)) lst
  |> Array.of_list


let timing fn msg =
  let c = 30 in
  let times = Owl.Utils.Stack.make () in
  for _ = 1 to c do
    let t = fn () in
    Owl.Utils.Stack.push times t
  done;
  let times = Owl.Utils.Stack.to_array times in
  let times = remove_outlier times in
  let m_time = Owl.Stats.mean times in
  let s_time = Owl.Stats.std times in
  Printf.printf "| %s :\t mean = %.5f \t std = %.5f \n%!" msg m_time s_time;
  m_time, s_time


let () =
  (* Build the network once. *)
  Owl_log.set_level FATAL;
  (* let f () =
   *   let g () = Model.detect () |> ignore in
   *   Owl_utils.time g
   * in
   * let _, _ = timing f "Graph Construction\n" in *)

  let f () =
    let fun_detect = Model.detect () in
    let g () = fun_detect src |> ignore in
    Owl_utils.time g
  in
  let _, _ = timing f "Mask R-CNN evaluation" in
  ()
