(* Performs object detection on each frame of a video and returns new video with
 * masks. *)
open Mrcnn

let video_file = "data/vids/video.mp4"

(* The next two functions are taken from
 * https://discuss.ocaml.org/t/how-to-create-a-temporary-directory-in-ocaml/
 * 1815/4 *)
let rand_digits () =
  let rand = Random.State.(bits (make_self_init ()) land 0xFFFFFF) in
  Printf.sprintf "%06x" rand

let mk_temp_dir ?(mode=0o700) pat =
  let dir = Filename.get_temp_dir_name () in
  let raise_err msg = raise (Sys_error msg) in
  let rec loop count =
    if count < 0 then raise_err "mk_temp_dir: too many failing attemps" else
    let dir = Printf.sprintf "%s/%s%s" dir pat (rand_digits ()) in
    try (Unix.mkdir dir mode; dir) with
    | Unix.Unix_error (Unix.EEXIST, _, _) -> loop (count - 1)
    | Unix.Unix_error (Unix.EINTR, _, _)  -> loop count
    | Unix.Unix_error (e, _, _)           ->
      raise_err ("mk_temp_dir: " ^ (Unix.error_message e))
  in
  loop 1000

let tmp_dir = mk_temp_dir "mrcnn_vid"
let img_file = tmp_dir ^ "/frame%04d.jpg"
let out_file = (Filename.dirname video_file) ^
                 "/output_" ^
                   (Filename.basename video_file)

let fun_detect = Model.detect ()

let cnt = ref 0

let eval_img src =
  cnt := !cnt + 1;
  Owl_log.info "Processing frame #%d..." !cnt;
  let Model.({rois; class_ids; masks; _}) = fun_detect src in
  if Array.length class_ids <> 0 then
    let img_arr = Image.img_to_ndarray src in
    (* add the bounding boxes and the masks to the picture *)
    Visualise.display_masks ~random_col:false img_arr rois masks class_ids;
    Image.save src Images.Jpeg (Image.img_of_ndarray img_arr)

let () =
  Owl_log.info
    "This script splits a complete video into frames and processes all the \
     frames one by one. Make sure the video is not too long and that it is OK \
     to use disk space to temporarily store frames.";
  let split_cmd = "ffmpeg -i " ^ video_file ^ " " ^ img_file ^ " -hide_banner"
  in
  let _ = Sys.command(split_cmd) in
  Array.iter (fun d -> eval_img (tmp_dir ^ "/" ^ d)) (Sys.readdir tmp_dir);
  let gather_cmd = "ffmpeg -framerate 30 -i " ^ img_file ^ " " ^ out_file in
  Sys.command(gather_cmd) |> ignore
