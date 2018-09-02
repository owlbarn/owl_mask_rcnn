open Owl
module AD = Neural.S.Algodiff
module N = Dense.Ndarray.S

module C = Configuration

(* *** Detection Layer *** *)
let refine_detections_graph =
  ()

let detection_layer () =
  fun inputs ->
    let inputs = Array.map AD.unpack_arr inputs in
    let rois = inputs.(0)
    and mrcnn_class = inputs.(1)
    and mrcnn_bbox = inputs.(2)
    and image_meta = inputs.(3) in

    let meta = Image.parse_image_meta image_meta in
    let image_shape = meta.image_shape in
    let h, w = N.get image_shape [|0;0|], N.get image_shape [|0;1|] in
    let window = Image.norm_boxes_graph meta.window [|h;w|] in

    let detections_batch = refine_detections_graph rois mrcnn_class mrcnn_bbox
                             window in
    let reshaped_detections = N.reshape detections_batch
                                [|C.batch_size; C.detection_max_instances|] in
    AD.pack_arr reshaped_detections

