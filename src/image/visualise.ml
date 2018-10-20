open Owl
module N = Dense.Ndarray.S

let apply_mask img mask colour =
  let alpha = 0.5 in
  N.iteri_nd (fun index x ->
      let i, j, k = index.(0), index.(1), index.(2) in
      if N.(mask.%{[|i; j|]}) >= 0.5 then
        N.set img index (x *. (1. -. alpha) +. alpha *. colour.(k)))
    img

let rnd () = float_of_int ((Random.int 180) + 10)

let random_colour () =
  [|rnd (); rnd (); rnd ()|]

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

let draw_contour img mask colour =
  let hm, wm = let s = N.shape img in s.(0) - 1, s.(1) - 1 in
  let is_contour x y =
    if N.get mask [|x; y|] <= 0.5 then false
    else if x <= 1 || y <= 1 || x >= hm - 1 || y >= wm - 1 then true
    else
      let contour = ref false in
      for i = x - 2 to x + 2 do
        for j = y - 2 to y + 2 do
          contour := !contour || N.get mask [|i; j|] <= 0.5;
        done;
      done;
      !contour
  in
  for i = 0 to hm do
    for j = 0 to wm do
      if is_contour i j then
        for k = 0 to 2 do
        N.set img [|i; j; k|] colour.(k);
        done;
    done;
  done

let col_by_class = Hashtbl.create 5

let display_masks ?(random_col=true) img boxes masks (class_ids : int array) =
  Random.self_init ();
  let n = (N.shape boxes).(0) in (* nb of instances *)
  for i = 0 to n - 1 do
    let mask = N.(get_slice [[i];[];[]] masks |> squeeze ~axis:[|0|]) in
    let box = N.(get_slice [[i];[]] boxes |> squeeze ~axis:[|0|]) in
    let colour =
      if random_col then random_colour ()
      else if Hashtbl.mem col_by_class class_ids.(i) then
        Hashtbl.find col_by_class class_ids.(i)
      else
        let col = random_colour () in
        Hashtbl.add col_by_class class_ids.(i) col;
        col
    in
    apply_mask img mask colour;
    draw_contour img mask colour;
    draw_box img box colour;
  done

let print_results class_ids boxes scores =
  Array.iteri (fun i id ->
      Printf.printf "%13s: %.3f " MrcnnUtil.class_names.(id)
        (N.get scores [|i|]);
      let y1, x1, y2, x2 =
        N.(int_of_float boxes.%{[|i; 0|]}, int_of_float boxes.%{[|i; 1|]},
           int_of_float boxes.%{[|i; 2|]}, int_of_float boxes.%{[|i; 3|]}) in
      Printf.printf "at [(%4d, %4d), (%4d, %4d)]\n" y1 x1 y2 x2)
    class_ids
