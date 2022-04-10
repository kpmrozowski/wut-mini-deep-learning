module Project2

using Base.Filesystem
using Flux
using Flux: @epochs, params, onecold, onehotbatch, MLUtils.DataLoader, logitcrossentropy, train!, glorot_uniform
using Random
using StableRNGs
using WAV

include("load.jl")
include("perceptron.jl")
include("utils.jl")

end # module
