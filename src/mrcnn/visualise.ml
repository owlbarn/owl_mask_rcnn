open Owl
module N = Dense.Ndarray.S

let apply_mask img mask colour =
  let alpha = 0.5 in
  N.iteri_nd (fun index x ->
      let i, j, k = index.(0), index.(1), index.(2) in
      if N.(mask.%{[|i; j|]}) >= 0.5 then
        N.set img index (x *. (1. -. alpha) +. alpha *. colour.(k)))
    img

let random_colours n =
  let rnd () = float_of_int ((Random.int 180) + 10) in
  Array.init n (fun _ -> [|rnd (); rnd (); rnd ()|])

let draw_hor_segment ?(width=2) img y x1 x2 colour =
  let h = (N.shape img).(0) in
  let bound = max 0 ((min h (y + width)) - 1) in
  for i = y to bound do
    for j = x1 to x2 - 1 do
      for k = 0 to 2 do
        N.set img [|i; j; k|] colour.(k);
      done;
    done;
  done

let draw_ver_segment ?(width=2) img x y1 y2 colour =
  let w = (N.shape img).(1) in
  let bound = max 0 ((min w (x + width)) - 1) in
  for i = y1 to y2 - 1 do
    for j = x to bound do
      for k = 0 to 2 do
        N.set img [|i; j; k|] colour.(k);
      done;
    done;
  done

let draw_box img box colour =
  let int_box = Array.map int_of_float
                  (Array.init 4 (fun i -> N.(box.%{[|i|]}))) in
  let y1, x1, y2, x2 = int_box.(0), int_box.(1), int_box.(2), int_box.(3) in
  draw_hor_segment img y1 x1 x2 colour;
  draw_hor_segment img y2 x1 x2 colour;
  draw_ver_segment img x1 y1 y2 colour;
  draw_ver_segment img x2 y1 y2 colour

let display_masks img boxes masks (* class_ids class_names score *) =
  Random.self_init ();
  let n = (N.shape boxes).(0) in (* nb of instances *)
  let colours = random_colours n in
  for i = 0 to n - 1 do
    let mask = N.(get_slice [[i];[];[]] masks |> squeeze ~axis:[|0|]) in
    let box = N.(get_slice [[i];[]] boxes |> squeeze ~axis:[|0|]) in
    let colour = colours.(i) in
    apply_mask img mask colour;
    draw_box img box colour;
  done
