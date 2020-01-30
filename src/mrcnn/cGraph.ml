module M = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.S)
module Compiler = Owl_neural_compiler.Make (M)

module Neural = Compiler.Neural
module Graph = Compiler.Neural.Graph
module AD = Compiler.Neural.Algodiff
module Engine = Compiler.Engine
