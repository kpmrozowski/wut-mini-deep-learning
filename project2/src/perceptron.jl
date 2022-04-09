function perceptron(seed::Integer)
    rng = StableRNG(seed)
    train = gettrain()

    small = filter(e -> e.label == :yes || e.label == :no, train)
    labels = [:yes, :no]
    train, val = splitdata(small, 0.1, rng)
    x_train = convert.(Float32, reduce(hcat, rpad.(getfield.(train, :data), 16000, 0)) ./ 2^15) |> gpu
    y_train = convert.(Float32, onehotbatch(getfield.(train, :label), labels)) |> gpu
    x_val = convert.(Float32, reduce(hcat, rpad.(getfield.(val, :data), 16000, 0)) ./ 2^15) |> gpu
    y_val = convert.(Float32, onehotbatch(getfield.(val, :label), labels)) |> gpu

    model = Chain(
        Dense(16000 => 800, relu, init=glorot_uniform(rng)),
        Dense(800 => 40, relu, init=glorot_uniform(rng)),
        Dense(40 => 2, init=glorot_uniform(rng)),
    ) |> gpu

    @epochs 100 begin
        train_loss = 0
        train_accuracy = 0

        train!(
            t -> begin
                out = model(t.data)
                loss = logitcrossentropy(out, t.label)
                train_loss += loss * size(t.data)[end] / length(train)
                pred = onecold(out, labels)
                corr = onecold(t.label, labels)
                train_accuracy += count(pred .== corr) / length(train)
                loss
            end,
            params(model),
            DataLoader((data=x_train, label=y_train), batchsize=32, shuffle=true, rng=rng),
            ADAM(),
        )

        out = model(x_val)
        val_loss = logitcrossentropy(out, y_val)
        pred = onecold(out, labels)
        corr = onecold(y_val, labels)
        val_accuracy = count(pred .== corr) / length(val)
        @info "Epoch training loss: $(train_loss)"
        @info "Epoch training accuracy: $(train_accuracy)"
        @info "Current validation loss: $(val_loss)"
        @info "Current validation accuracy: $(val_accuracy)"
    end
end
