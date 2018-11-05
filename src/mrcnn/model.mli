open Owl

module N = Dense.Ndarray.S

open CGraph.Graph

type results = {
    rois: N.arr;
    class_ids: int array;
    scores: N.arr;
    masks: int -> N.arr * int * int * int * int;
  }

val mrcnn : int -> network
(** [mrcnn num_anchors] returns the optimised Mask R-CNN network using
 ** parameters from the [configuration.ml] file. *)

val detect : unit -> string -> results
(** [detect ()] returns a function [string -> results] to evaluate the Mask
 ** R-CNN network on the given image file. *)
