open Owl
module N = Dense.Ndarray.S

let apply_mask img masks num color =
  let alpha = 0.5 in
  N.iteri_nd (fun index x ->
      let i, j, k = index.(0), index.(1), index.(2) in
      if N.(masks.%{[|num; i; j|]}) >= 0.5 then
        N.set img index (x *. (1. -. alpha) +. alpha *. color.(k)))
    img

let random_colors n =
  let rnd () = float_of_int ((Random.int 180) + 10) in
  Array.init n (fun _ -> [|rnd (); rnd (); rnd ()|])

let display_masks img boxes masks (* class_ids class_names score *) =
  Random.self_init ();
  let n = (N.shape boxes).(0) in (* nb of instances *)
  let colors = random_colors n in
  (* draw boxes? *)
  for i = 0 to n - 1 do
    apply_mask img masks i colors.(i);
  done
