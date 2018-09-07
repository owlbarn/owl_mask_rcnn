open Owl
open Neural.S
open Neural.S.Graph
module N = Dense.Ndarray.S

module PRA = PyramidROIAlign
module U = MrcnnUtil
module C = Configuration

(* *** Feature Pyramid Network *** *)
(* TODO: need TimeDistributed and PyramidROIAlign *)
let fpn_classifier_graph rois feature_maps meta
      pool_size num_classes fc_layers_size =
  let pyramid_fun = PRA.pyramid_roi_align [|pool_size; pool_size|] in
  let x =
    lambda_array [|C.post_nms_rois; pool_size; pool_size; 256|] pyramid_fun
      ~name:"roi_align_classifier" (Array.append [|rois; meta|] feature_maps)
    |> U.time_distributed (conv2d [|pool_size; pool_size; 256; fc_layers_size|]
                             [|1; 1|] ~padding:VALID ~name:"mrcnn_class_conv1")
    |> U.time_distributed (normalisation ~name:"mrcnn_class_bn1")
    |> activation Activation.Relu
    |> U.time_distributed (conv2d [|1; 1; fc_layers_size; fc_layers_size|]
                                [|1; 1|] ~padding:VALID ~name:"mrcnn_class_conv2")
    |> U.time_distributed (normalisation ~name:"mrcnn_class_bn2")
    |> activation Activation.Relu in

   (* squeeze dim 2 and 3?*)
  let shared = reshape [|C.post_nms_rois; fc_layers_size|]
                 ~name:"pool_squeeze" x in
  let mrcnn_class_logits =
    U.time_distributed (linear num_classes ~name:"mrcnn_class_logits") shared in
  let mrcnn_probs =
    U.time_distributed (activation Activation.(Softmax 1)
                          ~name:"mrcnn_class") mrcnn_class_logits in
  let x = U.time_distributed
            (linear (num_classes * 4) ~name:"mrcnn_bbox_fc") shared in

  let mrcnn_bbox = reshape [|C.post_nms_rois; num_classes; 4|] ~name:"mrcnn_bbox" x in
  mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

let build_fpn_mask_graph rois feature_maps image_meta pool_size num_classes =
  let pyramid_fun = PRA.pyramid_roi_align [|pool_size; pool_size|] in
  lambda_array [|C.detection_max_instances; pool_size; pool_size; 256|]
    pyramid_fun ~name:"roi_align_mask"
    (Array.append [|rois; image_meta|] feature_maps)
  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv1"
  |> normalisation ~name:"mrcnn_mask_bn1"
  |> activation Activation.Relu

  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv2"
  |> normalisation ~name:"mrcnn_mask_bn2"
  |> activation Activation.Relu

  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv3"
  |> normalisation ~name:"mrcnn_mask_bn3"
  |> activation Activation.Relu

  |> conv2d [|3; 3; 256; 256|] [|1; 1|] ~padding:SAME ~name:"mrcnn_mask_conv4"
  |> normalisation ~name:"mrcnn_mask_bn4"
  |> activation Activation.Relu

  |> transpose_conv2d [|2; 2; 256; 256|] [|2; 2|]
       ~act_typ:Activation.Relu ~name:"mrcnn_mask_deconv"
  |> conv2d [|1; 1; 256; num_classes|] [|1; 1|]
       ~act_typ:Activation.Sigmoid ~name:"mrcnn_mask"
