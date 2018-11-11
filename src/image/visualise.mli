open Owl

module N = Dense.Ndarray.S

val apply_mask : N.arr -> N.arr * int * int * int * int -> float array -> unit
(** [apply_mask img mask_xy colour] draws a mask of the specified colour on the
 ** image. [mask_xy] must be a binary array containing only 1's (where the
 ** object lies) and 0's (elsewhere). *)

val draw_box : N.arr -> N.arr -> float array -> unit
(** [draw_box img box colour] draws a rectangle [box] of the specified [colour]
 ** on [img].*)

val draw_contour : ?width:int -> N.arr -> N.arr * int * int * int * int
                   -> float array -> unit
(** [draw_contour img mask_xy colour] finds the edges of the mask and highlights
 ** them on [img] with the specified [colour]. *)

val display_labels : Images.t -> N.arr -> int array -> N.arr -> unit
(** [display_labels img boxes class_ids scores] writes the name of the class
 ** of each detected object on [img], along with the confidence [scores] of the
 ** detection. *)

val display_masks : ?random_col:bool -> N.arr -> N.arr
                    -> (int -> N.arr * int * int * int * int)
                    -> int array -> unit
(** [display_masks img boxes masks class_ids] applies the [masks], draws the
 ** [boxes] and highlights the edges of the binary masks. If [random_col] is
 ** false, systematically uses the same colour for elements of the same class,
 ** otherwise uses a random colour for each object. *)

val print_results : int array -> N.arr -> N.arr -> unit
(** [print_results class_ids boxes score] displays the class, confidence and
 ** position of each detected object. *)

val results_to_json : int array -> N.arr -> N.arr -> string
(** [results_to_json class_ids boxes scores] returns a json string with
 ** information about the detected objects (class, confidence and position). *)
