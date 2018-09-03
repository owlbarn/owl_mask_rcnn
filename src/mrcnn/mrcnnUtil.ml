open Owl
module N = Dense.Ndarray.S
module C = Configuration

let class_names =
  [|"BG"; "person"; "bicycle"; "car"; "motorcycle"; "airplane";
    "bus"; "train"; "truck"; "boat"; "traffic light";
    "fire hydrant"; "stop sign"; "parking meter"; "bench"; "bird";
    "cat"; "dog"; "horse"; "sheep"; "cow"; "elephant"; "bear";
    "zebra"; "giraffe"; "backpack"; "umbrella"; "handbag"; "tie";
    "suitcase"; "frisbee"; "skis"; "snowboard"; "sports ball";
    "kite"; "baseball bat"; "baseball glove"; "skateboard";
    "surfboard"; "tennis racket"; "bottle"; "wine glass"; "cup";
    "fork"; "knife"; "spoon"; "bowl"; "banana"; "apple";
    "sandwich"; "orange"; "broccoli"; "carrot"; "hot dog"; "pizza";
    "donut"; "cake"; "chair"; "couch"; "potted plant"; "bed";
    "dining table"; "toilet"; "tv"; "laptop"; "mouse"; "remote";
    "keyboard"; "cell phone"; "microwave"; "oven"; "toaster";
    "sink"; "refrigerator"; "book"; "clock"; "vase"; "scissors";
    "teddy bear"; "hair drier"; "toothbrush"|]

let compute_backbone_shapes image_shape strides =
  Array.init 5 (fun i ->
      Array.init 2 (fun j -> ceil (image_shape.(j) /. strides.(i))))

let generate_anchors scale ratios img_shape feature_stride anchor_stride =
  let ratios = N.of_array ratios [|(Array.length ratios)|] in
  let n = (N.shape ratios).(0) in
  let scale_arr = N.zeros (N.shape ratios) in
  N.fill scale_arr scale;
  let heights = N.((scale_arr / sqrt ratios) /$ 2.) in
  let widths = N.((scale_arr * sqrt ratios) /$ 2.) in

  let shifts_y, shifts_x =
    let nb_elts upper = (int_of_float ((upper -. 1.) /. anchor_stride)) + 1 in
    let build_shift s = N.sequential ~a:0. ~step:anchor_stride [|nb_elts s|] in
    N.(build_shift img_shape.(0) *$ feature_stride),
    N.(build_shift img_shape.(1) *$ feature_stride) in

  let ny = (N.shape shifts_y).(0)
  and nx = (N.shape shifts_x).(0) in
  let decomp x = ((x / (nx * n)) mod ny, (x / n) mod nx, x mod n) in
  let y1 = N.init [|ny * nx * n; 1|]
             (fun x -> let (i, _, k) = decomp x in
                       N.get shifts_y [|i|] -. N.get heights [|k|]) in
  let x1 = N.init [|ny * nx * n; 1|]
             (fun x -> let (_, j, k) = decomp x in
                       N.get shifts_x [|j|] -. N.get widths [|k|]) in
  let y2 = N.init [|ny * nx * n; 1|]
             (fun x -> let (i, _, k) = decomp x in
                       N.get shifts_y [|i|] +. N.get heights [|k|]) in
  let x2 = N.init [|ny * nx * n; 1|]
             (fun x -> let (_, j, k) = decomp x in
                       N.get shifts_x [|j|] +. N.get widths [|k|]) in
  let anchors = N.concatenate ~axis:1 [|y1; x1; y2; x2|] in
  anchors

let generate_pyramid_anchors scales ratios feature_shapes feature_strides
      anchor_stride =
  let anchors = Array.init (Array.length scales)
                  (fun i -> generate_anchors scales.(i) ratios feature_shapes.(i)
                              feature_strides.(i) anchor_stride) in
  N.concatenate ~axis:0 anchors

let get_anchors image_shape =
  let image_shape = Array.map float image_shape in
  let strides = Array.map float C.backbone_strides in
  let backbone_shapes = compute_backbone_shapes image_shape strides in
  let anchors = generate_pyramid_anchors C.rpn_anchor_scales C.rpn_anchor_ratios
                  backbone_shapes strides (float C.rpn_anchor_stride) in
  Image.norm_boxes anchors image_shape
