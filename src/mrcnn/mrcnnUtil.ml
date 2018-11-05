open Owl
module N = Dense.Ndarray.S
module C = Configuration


(* Names of the 80 classes of the MS Coco dataset. *)
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


let print_array arr =
  Printf.printf "[| ";
  Array.iter (fun i -> Printf.printf "%d " i) arr;
  Printf.printf "|]\n%!"


let print_array_float arr =
  Printf.printf "[| ";
  Array.iter (fun i -> Printf.printf "%0.3f " i) arr;
  Printf.printf "|]\n%!"


let comp2 (a, b) (c, d) =
  match compare a c with
  | 0 -> compare b d
  | x -> x


(* Returns an array with selected indices ix from array arr. *)
let gather_arr arr ix =
  Array.init (Array.length ix) (fun i -> arr.(ix.(i)))


(* Returns an array with selected nd-indices ix from ndarray t. *)
let gather_elts t ix =
  Array.init (Array.length ix) (fun i -> N.get t ix.(i))


(* Returns an Ndarray with selected nd-indices ix from ndarray t. *)
let gather_elts_nd t ix =
  N.init [|Array.length ix|] (fun i -> N.get t ix.(i))


(* Returns an Ndarray with selected indices ix from ndarray t. *)
let gather_elts_nd_arr t ix =
  N.init [|Array.length ix|] (fun i -> N.get t [|ix.(i)|])


(* Returns a list containing n empty lists (used for slicing). *)
let empty_lists n = List.init n (fun _ -> [])


(* Returns an Ndarray containing the selected slices with indices ix,
 * along the specified axis. *)
let gather_slice ?(axis=0) t ix =
  let dim = N.num_dims t in
  let first = empty_lists axis
  and last = empty_lists (dim - axis - 1) in
  let arr = Array.init (Array.length ix)
              (fun i -> N.get_slice (first @ ([[ix.(i)]] @ last)) t) in
  if Array.length arr > 0 then
    N.concatenate ~axis arr
  else (
    let sh = N.shape t in
    sh.(axis) <- 0;
    N.empty sh
  )


(* Creates an Ndarray with each slice initialised using the
 * slice: (int -> arr) function. *)
let init_slice ?(axis=0) shape slice =
  let dim = Array.length shape in
  let result = N.empty shape in
  let first = empty_lists axis
  and last = empty_lists (dim - axis - 1) in
  let end_of_loop = shape.(axis) - 1 in
  for i = 0 to end_of_loop do
    N.set_slice (first @ ([[i]] @ last)) result (slice i);
  done;
  result


(* Returns an array containing the indices between 0 and n - 1 satisfying
 * [cond i]. *)
let select_indices n cond =
  let rec loop i acc =
      if i >= n then List.rev acc
      else loop (i + 1) (if cond i then (i :: acc) else acc) in
  Array.of_list (loop 0 [])


(* Returns the unique elements from ids. *)
let unique_ids ids =
  let bitset = Array.make C.num_classes false in
  Array.iter (fun id -> bitset.(id) <- true) ids;
  select_indices C.num_classes (fun i -> bitset.(i))


(* Similar to Keras' TimeDistributed.*)
let time_distributed neuron input_node =
  let open CGraph.Graph in
  let open CGraph.AD in
  let input_shape = Neuron.get_out_shape input_node.neuron in
  let time_steps = input_shape.(0) in
  let dim = Array.length input_shape in
  let slice_shape = Array.sub input_shape 1 (dim - 1) in
  let x = lambda_array slice_shape
            (fun t ->
              let new_shape = Array.append [|(shape t.(0)).(0) * time_steps|]
                                slice_shape in
              Maths.reshape t.(0) new_shape) [|input_node|]
          |> neuron in
  let output_shape = Neuron.get_out_shape x.neuron
                     |> Array.append [|time_steps|] in
  lambda_array output_shape
    (fun t ->
      let batch_size = (shape t.(0)).(0) / time_steps in
      let final_shape = Array.append [|batch_size|] output_shape in
      Maths.reshape t.(0) final_shape) [|x|]


let pack t =
  CGraph.Engine.pack_arr t
  |> CGraph.AD.pack_arr


let unpack t =
  CGraph.AD.unpack_arr t
  |> CGraph.Engine.unpack_arr


(* Uses the LambdaArray node with a custom function defined with eager
 * evaluation. *)
let delay_lambda_array ?name shape f t =
  let exp_shape = Array.append [|1|] shape in
  CGraph.Graph.lambda_array ?name shape (fun t ->
      CGraph.AD.pack_arr (CGraph.M.delay_array exp_shape f
                            (Array.map CGraph.AD.unpack_arr t))) t
