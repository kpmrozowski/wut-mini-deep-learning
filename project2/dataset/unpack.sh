#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

cd "${0%/*}"

mkdir -p work
trap "rm -rf work" EXIT

bsdtar -xf "${1:-tensorflow-speech-recognition-challenge.zip}" --cd work
bsdtar -xf work/test.7z --cd work
bsdtar -xf work/train.7z --cd work

mv -T work/train .
mv -T work/test .
