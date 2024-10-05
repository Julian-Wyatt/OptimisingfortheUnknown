
#!/usr/bin/env bash

./build.sh $1

docker save cldetection_alg_2024 -o CLdetection_Alg_2024.tar
gzip CLdetection_Alg_2024.tar

