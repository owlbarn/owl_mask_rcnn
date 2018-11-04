open Owl

module N = Dense.Ndarray.S

module C = Configuration

(* *** ROIAlign Layer ***
 * Resizes all regions of interest from feature maps to a fixed size. *)

(* pool_shape: (pool_height, pool_width) of the output pool regions.
 * Returns: Pooled regions with shape
 * [batch_size, num_boxes, pool_height, pool_width, channels]. *)
let pyramid_roi_align pool_shape = fun inputs ->
  (* [batch, N, [y1, x1, y2, x2]] *)
  let boxes = inputs.(0) in
  (* [batch, (image data)] *)
  let image_meta = inputs.(1) in
  (* Different levels of feature maps from the backbone network. *)
  let feature_maps = Array.sub inputs 2 4 in

  let y1, x1, y2, x2 =
    let tmp = N.split ~axis:2 [|1;1;1;1|] boxes in
    tmp.(0), tmp.(1), tmp.(2), tmp.(3) in
  let h = N.(y2 - y1)
  and w = N.(x2 - x1) in
  (* Shape of the first image of batch, all the images have the same size. *)
  let image_shape = (Image.parse_image_meta image_meta).image_shape in
  let sqrt_image_area = sqrt (float (image_shape.(0) * image_shape.(1))) in
  let roi_level =
    let tmp = N.(log2 (sqrt (h * w) /$ (224. /. sqrt_image_area))) in
    let five = N.create [|1|] 5.
    and two = N.create [|1|] 2. in
    let roi_level = N.(min2 five (max2 two (round tmp +$ 4.))) in
    N.squeeze ~axis:[|2|] roi_level in

  let pooled = ref []
  and box_to_level = ref [] in
  for level = 2 to 5 do
    let i = level - 2 in
    let ix = N.filteri_nd (fun _ x -> int_of_float x = level)
               roi_level in
    if Array.length ix > 0 then (
      let level_boxes = N.init_nd [|Array.length ix; 4|]
                          (fun t -> N.get boxes [|0; ix.(t.(0)).(1); t.(1)|]) in
      box_to_level := ix :: !box_to_level;

      (* *** Stop gradient computation here if training the network *** *)

      pooled := (Image.crop_and_resize (N.squeeze ~axis:[|0|] feature_maps.(i))
                   level_boxes pool_shape) :: !pooled;
    )
  done;

  (* Rearranges pooled in the original order. *)
  let box_to_level =
    let tmp = Array.concat (List.rev !box_to_level) in
    let level_i = Array.init (Array.length tmp) (fun i -> (tmp.(i).(1), i)) in
    Array.sort MrcnnUtil.comp2 level_i;
    Array.init (Array.length level_i) (fun i -> snd level_i.(i)) in

  let pooled =
    let tmp = N.concatenate ~axis:0 (Array.of_list (List.rev !pooled)) in
    MrcnnUtil.gather_slice ~axis:0 tmp box_to_level in

  N.expand pooled 5
