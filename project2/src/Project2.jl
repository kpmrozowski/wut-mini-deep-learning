module Project2

using Base.Filesystem
using Flux
using Flux: @epochs, params, onehotbatch, MLUtils.DataLoader, logitcrossentropy, train!, glorot_uniform
using StableRNGs
using WAV

include("load.jl")
include("perceptron.jl")

end # module
