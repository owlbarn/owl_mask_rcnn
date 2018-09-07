open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
module N = Dense.Ndarray.S

module RPN = RegionProposalNetwork
module PL = ProposalLayer
module FPN = FeaturePyramidNetwork
module DL = DetectionLayer
module C = Configuration

(* Note: most function definitions only support batches of size 1, which is
 * enough for inference but not suitable for training. *)

(* *** MASK R-CNN *** *)
let mrcnn image_meta =
  let () =
    if C.image_shape.(0) mod 64 <> 0 || C.image_shape.(1) mod 64 <> 0 then
      invalid_arg "Image width and height must be divisible by 64" in
  (* compensates for the lack of Padding2D *)
  let input_shape = [|C.image_shape.(0) + 6; C.image_shape.(1) + 6; 3|] in
  let input_image = input ~name:"input_image" input_shape in

  (* The next two layers should be inputs of the network. Since Owl does not
   * support multi-inputs networks, we define them as input_image successors.
   * We can note that input_image_meta depends only on the size of the input
   * image. To avoid making the whole network dependent on the size of the
   * input, it has to be updated after the network is built. *)
  let f input t =
    let shape = Array.append [|(shape t.(0)).(0)|] (N.shape input) in
    pack_arr (N.reshape input shape) in

  let input_image_meta =
    lambda_array [|C.image_meta_size|] (f image_meta)
      ~name:"input_image_meta" [|input_image|] in
  let anchors = (* checked: the anchors are the same as the Keras ones *)
    let anchors = Image.get_anchors C.image_shape in
    lambda_array (N.shape anchors) (f anchors)
      ~name:"input_anchors" [|input_image|] in

  let _, c2, c3, c4, c5 = Resnet.resnet101 input_image in

  let tdps = C.top_down_pyramid_size in
  let p5 = conv2d [|1; 1; 2048; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c5p5" c5 in
  let p4 =
    add ~name:"fpn_p4add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5;
        conv2d [|1; 1; 1024; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c4p4" c4|] in
  let p3 =
    add ~name:"fpn_p3add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4;
        conv2d [|1; 1; 512; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c3p3" c3|] in

  let p2 =
    add ~name:"fpn_p2add"
      [|upsampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3;
        conv2d [|1; 1; 256; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c2p2" c2|] in

  let p2 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p2" p2 in
  let p3 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p3" p3 in
  let p4 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p4" p4 in
  let p5 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p5" p5 in
  let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in

  let rpn_feature_maps = [|p2; p3; p4; p5; p6|] in
  let mrcnn_feature_maps = [|p2; p3; p4; p5|] in

  let nb_ratios = Array.length C.rpn_anchor_ratios in
  (* it should be possible to create this network once and use it 5 times *)
  (* removed rpn_class_logits because useless for inference *)
  let rpns = Array.init 5
               (fun i -> RPN.rpn_graph rpn_feature_maps.(i)
                           nb_ratios C.rpn_anchor_stride tdps
                           ("_p" ^ string_of_int (i + 2))) in
  let rpn_class = concatenate 1 ~name:"rpn_class"
                    (Array.init 5 (fun i -> rpns.(i).(0))) in
  let rpn_bbox = concatenate 1 ~name:"rpn_bbox"
                   (Array.init 5 (fun i -> rpns.(i).(1))) in

  let rpn_rois =
    let prop_f = PL.proposal_layer C.post_nms_rois C.rpn_nms_threshold in
    lambda_array [|C.post_nms_rois; 4|] prop_f ~name:"ROI"
      [|rpn_class; rpn_bbox; anchors|] in

  let _, mrcnn_class, mrcnn_bbox =
    FPN.fpn_classifier_graph rpn_rois mrcnn_feature_maps input_image_meta
      C.pool_size C.num_classes C.fpn_classif_fc_layers_size in

  let detections = lambda_array [|C.detection_max_instances; 6|]
                    (DL.detection_layer ()) ~name:"mrcnn_detection"
                    [|rpn_rois; mrcnn_class; mrcnn_bbox; input_image_meta|] in
  let detection_boxes = lambda_array [|C.detection_max_instances; 4|]
                          (fun t ->
                            (* MrcnnUtil.print_array (shape t.(0)); *)
                            let x = Maths.get_slice [[]; []; [0;3]] t.(0) in
                            (* MrcnnUtil.print_array (shape x); *)
                            x)
                          [|detections|] in
  (*
  let mrcnn_mask = FPN.build_fpn_mask_graph detection_boxes mrcnn_feature_maps
                     input_image_meta C.mask_pool_size C.num_classes in *)
  get_network detection_boxes


(* *** Input and Output Processing *** *)

let update_image_meta nn image_meta =
  let input_meta_node = get_node nn "input_image_meta" in
  let image_meta = N.expand image_meta 2 in
  input_meta_node.output <- Some (pack_arr image_meta)

let get_output nn node_name =
  let node = get_node nn node_name in
  match node.output with
  | Some x -> unpack_arr x
  | None -> invalid_arg (node_name ^ " outputs have not been computed.")

(* Reformats the results of the neural network in a more suitable format.
 * detections: [N, [y1, x1, y2, x2, class_id, score]]
 * mrcnn_mask: [N, height, width, num_classes]
 * original_image_shape: [H, W, C]
 * image_shape: [H, W, C]
 * window: [y1, x1, y2, x2] *)
let unmold_detections detections mrcnn_mask original_image_shape image_shape
      window =
  (* Finds number of detections (detections is padded with zero but when valid,
   * class_id should be >= 1. Should be called on a single batch slice.
   * detections: *)
  let len = (N.shape detections).(0) in
  let n = let rec loop i =
            if i >= len then len
            else if N.get detections [|i; 4|] = 0. then i
            else loop (i + 1) in
          loop 0 in

  let boxes = N.get_slice [[0;n-1];[0;3]] detections in
  let window = Image.norm_boxes
                 (N.of_array (Array.map float window) [|4|]) image_shape in
  let wy1, wx1, wy2, wx2 =
    N.(window.%{[|0|]}, window.%{[|1|]}, window.%{[|2|]}, window.%{[|3|]}) in
  let shift = N.of_array [|wy1; wx1; wy1; wx1|] [|4|] in
  let wh = wy2 -. wy1
  and ww = wx2 -. wx1 in
  let scale = N.of_array [|wh; ww; wh; ww|] [|4|] in
  let boxes =
    let tmp_boxes = N.((boxes - shift) / scale) in
    Image.denorm_boxes tmp_boxes original_image_shape in

  (* Keep only boxes with area > 0 *)
  let keep = MrcnnUtil.select_indices n (fun i ->
                 N.((boxes.%{[|i;2|]} -. boxes.%{[|i;0|]}) *.
                      (boxes.%{[|i;3|]} -. boxes.%{[|i;1|]})) > 0.) in
  let n = Array.length keep in
  let boxes = MrcnnUtil.gather_slice ~axis:0 boxes keep in
  (* Extracts boxes, class_ids, scores and masks *)
  let class_ids = Array.init n (fun i ->
                      int_of_float (N.get detections [|keep.(i); 4|])) in
  let masks =
    let mask_h, mask_w = let sh = N.shape mrcnn_mask in sh.(1), sh.(2) in
    MrcnnUtil.init_slice ~axis:0 [|n; mask_h; mask_w|]
      (fun i -> N.get_slice [[keep.(i)];[];[];[class_ids.(i)]] mrcnn_mask
                |> N.squeeze ~axis:[|3|]) in
  let scores = MrcnnUtil.gather_slice ~axis:0
                 (N.get_slice [[];[5]] detections) keep in

  let full_masks =
    let h, w = original_image_shape.(0), original_image_shape.(1) in
    MrcnnUtil.init_slice ~axis:0 [|n; h; w|] (fun i ->
        let mask = N.get_slice [[i];[];[]] masks |> N.squeeze ~axis:[|0|] in
        let box = N.get_slice [[i];[];[]] boxes |> N.squeeze ~axis:[|0|] in
        Image.unmold_mask mask box original_image_shape) in

  boxes, class_ids, scores, full_masks

type results = {
    rois: N.arr;
    class_ids: int array;
    scores: N.arr;
    masks: N.arr;
  }

let extract_features nn image_meta =
  let detections = get_output nn "mrcnn_detection" in
  let mrcnn_masks = get_output nn "mrcnn_mask" in
  let meta = Image.parse_image_meta image_meta in

  let rois, class_ids, scores, masks =
    unmold_detections
      (N.squeeze ~axis:[|0|] detections)
      (N.squeeze ~axis:[|0|] mrcnn_masks)
      meta.original_image_shape
      meta.image_shape
      meta.window in
  { rois; class_ids; scores; masks; }


let detect src =
  let molded_image, image_meta, windows = Image.mold_inputs src in
  let nn = mrcnn image_meta in (* depends only on the size of the image... *)
  Graph.init nn;
  (* Graph.load_weights nn weight_file; *)
  (* Graph.print nn; *)

  (fun () ->
    (* quick hack to replace zero_padding2d *)
    let image = N.pad ~v:0. [[3;3];[3;3];[0;0]] molded_image in
    let image = N.expand image 4 in

    let t = Graph.model nn image in
    t
    (*
    let results = extract_features nn image_meta in
    results *)
  )
