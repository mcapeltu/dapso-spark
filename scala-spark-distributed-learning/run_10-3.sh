#!/usr/bin/env bash

set -euo pipefail
export SPARK_LOCAL_IP=127.0.0.1

mkdir -p results/repetitions/dataset_scaling
mkdir -p results/logs/dataset_scaling

for i in $(seq -w 1 10); do
    echo "===== Dataset-scaling run $i/10 ====="

    rm -f resultados_pso.txt resultados_dapso.txt

    sbt run 2>&1 | tee "results/logs/dataset_scaling/run_${i}.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
        echo "ERROR: run $i failed" >&2
        exit "$status"
    fi

    run_dir="results/repetitions/dataset_scaling/run_${i}"
    mkdir -p "$run_dir"

    if [ ! -f resultados_pso.txt ] || [ ! -f resultados_dapso.txt ]; then
        echo "ERROR: run $i did not generate the expected result files" >&2
        exit 1
    fi

    mv resultados_pso.txt resultados_dapso.txt "$run_dir/"

    echo "Run $i completed and stored in $run_dir"
done

echo "All dataset-scaling runs completed."
