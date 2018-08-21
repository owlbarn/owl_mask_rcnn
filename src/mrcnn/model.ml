module C = Configuration

open Owl
open Owl_types
open Neural 
open Neural.S
open Neural.S.Graph
module AD = Owl.Algodiff
module N = Dense.Ndarray.S

(* *** RESNET101 ***
 * The code is heavily inspired by
 * https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py *)

let id_block input kernel_size filters stage block input_layer =
  let suffix = string_of_int stage ^ block ^ "_branch" in
  let conv_name = "res" ^ suffix in
  let bn_name = "bn" ^ suffix in
  let f1, f2, f3 = filters in
  let x =
    input_layer
    |> conv2d [|1; 1; input; f1|] [|1; 1|] ~padding:VALID ~name:(conv_name^"2a")
    |> normalisation ~axis:3 ~name:(bn_name^"2a") (* 3 should be the axis since [|1;224;224;3|] *)
    |> activation Activation.Relu
                  
    |> conv2d [|kernel_size; kernel_size; f1; f2|] [|1; 1|] ~padding:SAME ~name:(conv_name^"2b")
    |> normalisation ~axis:3 ~name:(bn_name^"2b")
    |> activation Activation.Relu
                  
    |> conv2d [|1; 1; f2; f3|] [|1; 1|] ~padding:VALID ~name:(conv_name^"2c")
    |> normalisation ~axis:3 ~name:(bn_name^"2c") in

  add [|x; input_layer|]
  |> activation Activation.Relu

let conv_block input kernel_size filters strides stage block input_layer =
  let suffix = string_of_int stage ^ block ^ "_branch" in
  let conv_name = "res" ^ suffix in
  let bn_name = "bn" ^ suffix in
  let f1, f2, f3 = filters in
  let x =
    input_layer
    |> conv2d [|1; 1; input; f1|] strides ~padding:VALID ~name:(conv_name^"2a")
    |> normalisation ~axis:3 ~name:(bn_name^"2a")
    |> activation Activation.Relu
                  
    |> conv2d [|kernel_size; kernel_size; f1; f2|] [|1; 1|] ~padding:SAME ~name:(conv_name^"2b")
    |> normalisation ~axis:3 ~name:(bn_name^"2b")
    |> activation Activation.Relu
                  
    |> conv2d [|1; 1; f2; f3|] [|1; 1|] ~padding:VALID ~name:(conv_name^"2c")
    |> normalisation ~axis:3 ~name:(bn_name^"2c") in
  
  let shortcut =
    input_layer
    |> conv2d [|1; 1; input; f3|] strides ~name:(conv_name^"1")
    |> normalisation ~axis:3 ~name:(bn_name^"1") in

  add [|x; shortcut|]
  |> activation Activation.Relu

let resnet101 input_image =
  (* +6 is a quick hack instead of zero_padding2d [|3; 3|] *)
  let c1 = 
    input_image
    (* should be |> zero_padding2d [|3; 3|] ~name:"conv1_pad" *)
    |> conv2d [|7; 7; 3; 64|] [|2; 2|] ~padding:VALID ~name:"conv1"
    |> normalisation ~axis:3 ~name:"bn_conv1"
    |> activation Activation.Relu
    |> max_pool2d [|3; 3|] [|2; 2|] in
  let c2 =
    conv_block 64 3 (64, 64, 256) [|1; 1|] 2 "a" c1
    |> id_block 256 3 (64, 64, 256) 2 "b"
    |> id_block 256 3 (64, 64, 256) 2 "c" in

  let c3 =
    conv_block 256 3 (128, 128, 512) [|2; 2|] 3 "a" c2
    |> id_block 512 3 (128, 128, 512) 3 "b"
    |> id_block 512 3 (128, 128, 512) 3 "c"
    |> id_block 512 3 (128, 128, 512) 3 "d" in
                
  let x =
    conv_block 512 3 (256, 256, 1024) [|2; 2|] 4 "a" c3 in
  let y = ref x in
  for i = 0 to 21 do  (* code('b') is 98 *)
    y := id_block 1024 3 (256, 256, 1024) 4 (Char.escaped (Char.chr (98 + i))) !y
  done;
  let c4 = !y in
  
  let c5 =
    conv_block 1024 3 (512, 512, 2048) [|2; 2|] 5 "a" c4
    |> id_block 2048 3 (512, 512, 2048) 5 "b"
    |> id_block 2048 3 (512, 512, 2048) 5 "c" in
  (c1, c2, c3, c4, c5)

(* *** REGION PROPOSAL NETWORK *** *)
let rpn_graph feature_map anchors_per_location anchor_stride =
  let shared = conv2d [|3; 3; 256; 512|] [|anchor_stride; anchor_stride|] (* not 256 *)
                 ~padding:SAME ~act_typ:Activation.Relu ~name:"rpn_conv_shared"
                 feature_map in
  let x = conv2d [|1; 1; 512; 2 * anchors_per_location|] [|1; 1|]
            ~padding:VALID ~name:"rpn_class_raw" shared in
  let rpn_class_logits = lambda (fun t -> N.reshape t [|(N.shape t).(0); -1; 2|]) x in
  let rpn_probs = activation Activation.(Softmax 1)
                    ~name:"rpn_class_xxx" rpn_class_logits in
  let x = conv2d [|1; 1; 512; anchors_per_location * 4|] [|1; 1|] ~padding:VALID
            ~name:"rpn_bbox_pred" shared in
  let rpn_bbox = lambda (fun t -> N.reshape t [|(N.shape t).(0); -1; 4|]) x in
  [|rpn_class_logits; rpn_probs; rpn_bbox|]
                 
let build_rpn_model anchor_stride anchors_per_location depth =
  let input_feature_map = input [|256; 256; depth|] ~name:"input_rpn_feature_map" in (* not 256 *)
  let outputs = rpn_graph input_feature_map anchors_per_location anchor_stride in
  get_network ~name:"rpn_model" (outputs input_feature_map)
    
(* *** MASK R-CNN *** *)
let mrcnn () =
  let input_image = input ~name:"input_image" C.image_shape in
  let input_image_meta = input ~name:("input_image_meta") [|C.image_meta_size|] in
  let anchors = input ~name:"input_anchors" [|256; 4|] in (* 256? How many anchors?*)
  let _, c2, c3, c4, c5 = resnet101 input_image in
  
  let tdps = C.top_down_pyramid_size in
  let p5 = conv2d [|1; 1; 2048; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c5p5" c5 in
  (* change this after you have upsampling2d *)
  let p4 =
    add ~name:"fpn_p4add"
      [|p5; (* up_sampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5 *) 
        conv2d [|1; 1; 1024; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c4p4" c4|] in
  let p3 =
    add ~name:"fpn_p3add"
      [|p4; (* up_sampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4 *)
        conv2d [|1; 1; 512; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c3p3" c3|] in
  let p2 =
    add ~name:"fpn_p2add"
      [|p3; (* up_sampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3 *)
        conv2d [|1; 1; 256; tdps|] [|1; 1|] ~padding:VALID ~name:"fpn_c2p2" c2|] in
  let p2 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p2" p2 in
  let p3 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p3" p3 in
  let p4 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p4" p4 in
  let p5 = conv2d [|3; 3; tdps; tdps|] [|1; 1|] ~padding:SAME ~name:"fpn_p5" p5 in

  let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in
  p2
  
