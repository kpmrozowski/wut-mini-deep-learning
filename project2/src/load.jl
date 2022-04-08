function loadone(name::AbstractString)
    data, f = wavread(name, format="native")
    @assert f == 16000
    vec(data)
end

loaddir(dir::AbstractString) = cd(joinpath(@__DIR__, "..", "dataset")) do
    map(loadone, filter(s -> endswith(s, ".wav"), readdir(dir, join=true, sort=true)))
end

dirs2labels = ["yes" => :yes, "no" => :no, "up" => :up, "down" => :down, "left" => :left, "right" => :right, "on" => :on, "off" => :off, "stop" => :stop, "go" => :go, "bed" => :unknown, "bird" => :unknown, "cat" => :unknown, "dog" => :unknown, "eight" => :unknown, "five" => :unknown, "four" => :unknown, "happy" => :unknown, "house" => :unknown, "marvin" => :unknown, "nine" => :unknown, "one" => :unknown, "seven" => :unknown, "sheila" => :unknown, "six" => :unknown, "three" => :unknown, "tree" => :unknown, "two" => :unknown, "wow" => :unknown, "zero" => :unknown]

gettrain() = reduce(vcat, map(pair -> map(data -> data => pair.second, loaddir("train/audio/$(pair.first)")), dirs2labels))
getbgnoise() = loaddir("train/audio/_background_noise_")
gettest() = loaddir("test/audio")
