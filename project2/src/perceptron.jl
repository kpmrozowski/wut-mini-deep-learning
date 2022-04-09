function perceptron(seed::Integer)
    rng = StableRNG(seed)
    train = gettrain()

    yes = filter(x -> x.label == :yes, train)
    no = filter(x -> x.label == :no, train)

    x_train = convert.(Float32, reduce(hcat, rpad.(getfield.(vcat(yes, no), :data), 16000, 0))) |> gpu
    y_train = convert.(Float32, onehotbatch(getfield.(vcat(yes, no), :label), [:yes, :no])) |> gpu

    model = Chain(
        Dense(16000 => 800, relu, init=glorot_uniform(rng)),
        Dense(800 => 40, relu, init=glorot_uniform(rng)),
        Dense(40 => 2, init=glorot_uniform(rng))
    ) |> gpu

    @epochs 3 train!(
        t -> logitcrossentropy(model(t.data), t.label),
        params(model),
        DataLoader((data=x_train, label=y_train), batchsize=32, shuffle=true, rng=rng),
        ADAM(),
    )
end
