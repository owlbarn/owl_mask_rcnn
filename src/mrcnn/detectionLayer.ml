open Owl
module AD = Neural.S.Algodiff
module N = Dense.Ndarray.S

module C = Configuration

(* *** Detection Layer *** *)

let refine_detections rois probs deltas window =
  let n = (N.shape rois).(0) in
  let class_ids = Array.init n
                    (fun i -> (snd (N.max_i (N.get_slice [[i];[]] rois))).(1)) in
  let indices = Array.init n
                  (fun i -> [|i; class_ids.(i)|]) in
  let class_scores = N.get_index probs indices in
  let deltas_specific =
    N.init_nd [|n; 4|]
      (fun t -> N.get deltas [|t.(0); class_ids.(t.(0)); t.(1)|]) in
  let refined_rois =
    let delta_rois = Image.apply_box_deltas rois
                       N.(deltas_specific * C.bbox_std_dev) in
    Image.clip_boxes delta_rois window in

  (* Array of indices of the non background boxes with confidence over the
   * threshold. *)
  let keep =
    let cond i = class_ids.(i) > 0 &&
                   class_scores.(i) > C.detection_min_confidence in
    MrcnnUtil.select_indices n cond in

  (* Per class NMS *)
  let pre_nms_class_ids = MrcnnUtil.gather_arr class_ids keep in
  let pre_nms_scores = MrcnnUtil.gather_arr class_scores keep in
  let pre_nms_rois = MrcnnUtil.gather_slice ~axis:0 refined_rois keep in
  let unique_pre_nms_class_ids = MrcnnUtil.unique_ids pre_nms_class_ids in

  let nms_keep_map class_id =
    let ixs = MrcnnUtil.select_indices (Array.length pre_nms_class_ids)
                (fun i -> class_id = pre_nms_class_ids.(i)) in
    let class_keep =
      Image.non_max_suppression
        (MrcnnUtil.gather_slice ~axis:0 pre_nms_rois ixs)
        (N.of_array (MrcnnUtil.gather_arr pre_nms_scores ixs) [|Array.length ixs|])
        C.detection_max_instances C.detection_nms_threshold in
    MrcnnUtil.gather_arr keep (MrcnnUtil.gather_arr ixs class_keep) in
    (*Array.init C.detection_max_instances
      (fun i -> if i < Array.length class_keep then class_keep.(i) else -1) in*)

  let nms_keep =
    let tmp = Array.map nms_keep_map unique_pre_nms_class_ids in
    (* Not super efficient but should be a really small list *)
    Array.concat (Array.to_list tmp) in

  (* The Keras implementation computes here the intersection between keep and
   * nms_keep but I am pretty sure that this is useless. *)

  (* Should I sort them by score? *)

  let detections =
    let refined_rois_keep = MrcnnUtil.gather_slice refined_rois nms_keep in
    let class_scores_keep = MrcnnUtil.gather_arr class_scores nms_keep in
    let class_ids_keep = MrcnnUtil.gather_arr class_ids nms_keep in
    let slice i =
      N.init [|1; 6|] (fun j ->
          if j < 4 then N.get refined_rois_keep [|i;j|]
          else if j = 4 then float class_ids_keep.(i)
          else class_scores_keep.(i)) in
    MrcnnUtil.init_slice ~axis:0 [|Array.length nms_keep; 6|] slice in

  let pad_bottom = C.detection_max_instances - (N.shape detections).(0) in
  N.pad ~v:0. [[0;pad_bottom]; [0;0]] detections


let detection_layer () = fun inputs ->
  let inputs = Array.map AD.unpack_arr inputs in
  let rois = inputs.(0)
  and mrcnn_class = inputs.(1)
  and mrcnn_bbox = inputs.(2)
  and image_meta = inputs.(3) in

  let meta = Image.parse_image_meta image_meta in
  let image_shape = meta.image_shape in
  let window = Array.map float (meta.window) in
  let h, w = image_shape.(0), image_shape.(1) in
  let window = Image.norm_boxes (N.of_array window [|4|]) [|h; w|] in

  let detections_batch = refine_detections rois mrcnn_class mrcnn_bbox window in
  let reshaped_detections = N.reshape detections_batch
                              [|C.batch_size; C.detection_max_instances; 6|] in
  AD.pack_arr reshaped_detections
