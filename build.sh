#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t cldetection_alg_2024 "$SCRIPTPATH" -f $1 --progress=plain