open Owl

module N = Dense.Ndarray.S


let apply_mask img mask_xy colour =
  let mask, y1, x1, y2, x2 = mask_xy in
  let mask = N.expand ~hi:true mask 3 in
  let alpha = 0.4 in
  for k = 0 to 2 do
    let slice = [[y1; y2 - 1]; [x1; x2 - 1]; [k]] in
    let rect = N.get_slice slice img in
    let to_set = N.copy rect in
    N.sub_scalar_ rect colour.(k);
    N.mul_scalar_ rect alpha;
    N.mul_ ~out:rect rect mask;
    N.sub_ ~out:to_set to_set rect;
    N.set_slice slice img to_set
  done


let rnd_char () = float_of_int ((Random.int 160) + 30)


let random_colour () = [|rnd_char (); rnd_char (); rnd_char ()|]


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
  let int_box = Array.init 4 (fun i -> int_of_float N.(box.%{[|i|]})) in
  let y1, x1, y2, x2 = int_box.(0), int_box.(1), int_box.(2), int_box.(3) in
  draw_hor_segment img y1 x1 x2 colour;
  draw_hor_segment img y2 x1 x2 colour;
  draw_ver_segment img x1 y1 y2 colour;
  draw_ver_segment img x2 y1 y2 colour


let draw_contour ?(width=2) img mask_xy colour =
  let mask, y1, x1, y2, x2 = mask_xy in
  let ym, xm = y2 - 1, x2 - 1 in
  let is_contour y x =
    if N.get mask [|y - y1; x - x1|] = 0. then false
    else if y <= y1 + width - 1 || x <= x1 + width - 1 ||
              y >= ym - width + 1 || x >= xm - width + 1 then true
    else
      let contour = ref false in
      for i = y - width to y + width do
        for j = x - width to x + width do
          contour := !contour || N.get mask [|i - y1; j - x1|] = 0.;
        done;
      done;
      !contour
  in
  for i = y1 to ym do
    for j = x1 to xm do
      if is_contour i j then
        for k = 0 to 2 do
          N.set img [|i; j; k|] (min 255. (colour.(k) +. 30.));
        done;
    done;
  done


let draw_text =
  let face = new OFreetype.face "FreeMonoBold.ttf" 0 in
  fun img box text ->
  let y1, x1 = N.(int_of_float box.%{[|0|]}, int_of_float box.%{[|1|]}) in
  let text = Fttext.unicode_of_latin text in
  let size = max ((max img#width img#height) / 120 + 1) 10 in
  face#set_char_size 100. 100. size size;
  OFreetype.draw_text face (fun org level ->
      let level' = 255 - level in
      Images.{ r = (org.r * level' + 255 * level) / 255;
               g = (org.g * level' + 255 * level) / 255;
               b = (org.b * level' + 255 * level) / 255 }) img x1 y1 text


let display_labels img boxes class_ids scores =
  let img = match img with
    | Images.Rgb24 img -> img
    | _ -> failwith "not implemented format"
  in
  let n = (N.shape boxes).(0) in
  let img24 = new OImages.rgb24_wrapper img in
  for i = 0 to n - 1 do
    let box = N.(get_slice [[i];[]] boxes |> squeeze ~axis:[|0|]) in
    let text = Printf.sprintf "%s %.2f" MrcnnUtil.class_names.(class_ids.(i))
                 (N.get scores [|i|]) in
    draw_text (img24 :> Color.rgb OImages.map) box text;
  done


(* Stores the colour used for each class. *)
let col_by_class : (int, float array) Hashtbl.t = Hashtbl.create 5


let display_masks ?(random_col=true) img boxes masks class_ids =
  Random.self_init ();
  let n = (N.shape boxes).(0) in (* nb of instances *)
  for i = 0 to n - 1 do
    let mask = masks i in
    let box = N.(get_slice [[i];[]] boxes |> squeeze ~axis:[|0|]) in
    let colour =
      if random_col then random_colour ()
      else if Hashtbl.mem col_by_class class_ids.(i) then
        Hashtbl.find col_by_class class_ids.(i)
      else (
        let col = random_colour () in
        Hashtbl.add col_by_class class_ids.(i) col;
        col
      )
    in
    apply_mask img mask colour;
    draw_box img box colour;
    draw_contour img mask colour;
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


let results_to_json class_ids boxes scores =
  let n = Array.length class_ids in
  let b = Buffer.create 150 in
  Buffer.add_char b '[';
  let json_arr = Array.init n
    (fun i ->
      let y1, x1, y2, x2 =
        N.(int_of_float boxes.%{[|i; 0|]}, int_of_float boxes.%{[|i; 1|]},
           int_of_float boxes.%{[|i; 2|]}, int_of_float boxes.%{[|i; 3|]}) in
      Printf.sprintf "{\"class\":\"%13s\", \"conf\":\"%.3f\",\
                      \"pos\":\"(%4d, %4d), (%4d, %4d)\"}"
      MrcnnUtil.class_names.(class_ids.(i)) (N.get scores [|i|]) y1 x1 y2 x2)
  in
  Array.iteri (fun i s ->
      Buffer.add_string b s;
      if i <> n - 1 then Buffer.add_char b ',') json_arr;
  Buffer.add_char b ']';
  Buffer.contents b
