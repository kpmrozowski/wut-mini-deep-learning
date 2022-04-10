function splitdata(data::AbstractArray, percentval::Number, rng::StableRNG)
    pivot = floor(Int, length(data) * percentval)
    perm = randperm(rng, length(data))
    valind = sort(perm[1:pivot])
    trainind = sort(perm[(pivot + 1):end])
    (train = data[trainind], val = data[valind])
end
