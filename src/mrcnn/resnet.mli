open CGraph.Graph

val resnet101 : node -> node * node * node * node * node
(** [resnet101 input_layer] returns 5 nodes from the ResNet101 architecture,
 ** from the most low-level to the most high-level. *)
