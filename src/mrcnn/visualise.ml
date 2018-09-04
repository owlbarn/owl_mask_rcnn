open Owl
module N = Dense.Ndarray.S

let apply_mask img mask color =
  let alpha = 0.5 in
  N.iteri_nd (fun index t ->
      let i, j, k = index.(0), index.(1), index.(2) in
      if mask.(i).(j) = 1 then
        N.set img index (t *. (1. -. alpha) +. alpha *. color.(k) *. 255.))
    img

let random_colors n =
  Random.self_init ();
  let rnd () = float_of_int ((Random.int 200) + 55) in
  Array.init n (fun _ -> [|rnd (); rnd (); rnd ()|])

(* Not completed yet. *)
let display_masks img boxes masks class_ids class_names score =
  let n = (N.shape boxes).(0) in (* nb of instances *)
  let colors = random_colors n in
  for i = 0 to n - 1 do
    apply_mask img masks.(i) colors.(i);
  done;
