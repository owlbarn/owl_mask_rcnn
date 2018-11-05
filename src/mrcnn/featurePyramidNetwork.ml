open Owl
open CGraph.Neural
open CGraph.Graph

module PRA = PyramidROIAlign
module U = MrcnnUtil
module C = Configuration

(* *** FEATURE PYRAMID NETWORK ***
 * Associates a probability to each class for each proposal and refines the
 * bounding box for each class even more. *)

(* rois: [batch, N, (y1, x1, y2, x2)],
 * feature_maps: feature maps from different levels of the backbone,
 * image_meta: [batch, (image info)],
 * pool_size: dimension of the square feature map of ROIAlign,
 * num_classes: number of different classes,
 * fc_layers_size: size of the fully connected layers.
 * Returns:
 * mrcnn_probs: [batch, number of ROIs, number of classes]
 * mrcnn_bbox: [batch, number of ROIs, num classes, (dx, dy, log(dh), log(dw))]
 *             further bounding box refinement for each class. *)
let fpn_classifier_graph rois feature_maps meta
      pool_size num_classes fc_layers_size =
  let pyramid_fun = PRA.pyramid_roi_align [|pool_size; pool_size|] in
  let x =
    MrcnnUtil.delay_lambda_array
      [|C.post_nms_rois; pool_size; pool_size; 256|] pyramid_fun
      ~name:"roi_align_classifier" (Array.append [|rois; meta|] feature_maps)
    |> U.time_distributed (conv2d [|pool_size; pool_size; 256; fc_layers_size|]
                             [|1; 1|] ~padding:VALID ~name:"mrcnn_class_conv1")
    |> U.time_distributed (normalisation ~name:"mrcnn_class_bn1")
    |> activation Activation.Relu
    |> U.time_distributed
         (conv2d [|1; 1; fc_layers_size; fc_layers_size|]
            [|1; 1|] ~padding:VALID ~name:"mrcnn_class_conv2")
    |> U.time_distributed (normalisation ~name:"mrcnn_class_bn2")
    |> activation Activation.Relu in

  let shared = reshape [|C.post_nms_rois; fc_layers_size|]
                 ~name:"pool_squeeze" x in
  let mrcnn_class_logits =
    U.time_distributed (linear num_classes ~name:"mrcnn_class_logits") shared in
  let mrcnn_probs =
    U.time_distributed (activation Activation.(Softmax 1)
                          ~name:"mrcnn_class") mrcnn_class_logits in
  let x = U.time_distributed
            (linear (num_classes * 4) ~name:"mrcnn_bbox_fc") shared in

  let mrcnn_bbox = reshape [|C.post_nms_rois; num_classes; 4|]
                     ~name:"mrcnn_bbox" x in
  (* mrcnn_class_logits is useless in inference mode *)
  (* mrcnn_class_logits, *) mrcnn_probs, mrcnn_bbox


(* rois: [batch, N, (y1, x1, y2, x2),
 * feature_maps: feature maps from different levels of the backbone,
 * image_meta: [batch, (image info)],
 * pool_size: dimension of the square feature map of ROIAlign,
 * num_classes: number of different classes,
 * fc_layers_size: size of the fully connected layers.
 * Returns:
 * masks: [batch, number of ROIs, pool_size, pool_size, num_classes]. *)
let build_fpn_mask_graph rois feature_maps image_meta pool_size num_classes =
  let pyramid_fun = PRA.pyramid_roi_align [|pool_size; pool_size|] in
  MrcnnUtil.delay_lambda_array
    [|C.detection_max_instances; pool_size; pool_size; 256|]
    pyramid_fun ~name:"roi_align_mask"
    (Array.append [|rois; image_meta|] feature_maps)
  |> U.time_distributed (conv2d [|3; 3; 256; 256|] [|1; 1|]
                           ~padding:SAME ~name:"mrcnn_mask_conv1")
  |> U.time_distributed (normalisation ~name:"mrcnn_mask_bn1")
  |> activation Activation.Relu

  |> U.time_distributed (conv2d [|3; 3; 256; 256|] [|1; 1|]
                           ~padding:SAME ~name:"mrcnn_mask_conv2")
  |> U.time_distributed (normalisation ~name:"mrcnn_mask_bn2")
  |> activation Activation.Relu

  |> U.time_distributed (conv2d [|3; 3; 256; 256|] [|1; 1|]
                           ~padding:SAME ~name:"mrcnn_mask_conv3")
  |> U.time_distributed (normalisation ~name:"mrcnn_mask_bn3")
  |> activation Activation.Relu

  |> U.time_distributed (conv2d [|3; 3; 256; 256|] [|1; 1|]
                           ~padding:SAME ~name:"mrcnn_mask_conv4")
  |> U.time_distributed (normalisation ~name:"mrcnn_mask_bn4")
  |> activation Activation.Relu

  |> U.time_distributed
       (transpose_conv2d [|2; 2; 256; 256|] [|2; 2|] ~padding:VALID
          ~act_typ:Activation.Relu ~name:"mrcnn_mask_deconv")
  |> U.time_distributed
       (conv2d [|1; 1; 256; num_classes|] [|1; 1|] ~padding:VALID
          ~act_typ:Activation.Sigmoid ~name:"mrcnn_mask")
