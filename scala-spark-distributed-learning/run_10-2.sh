#!/usr/bin/env bash

set -euo pipefail

mkdir -p results/repetitions/regression
mkdir -p results/logs/regression

for i in $(seq -w 1 10); do
    echo "===== Regression run $i/10 ====="

    rm -f resultados_reg_*.txt

    sbt run 2>&1 | tee "results/logs/regression/run_${i}.log"

    run_dir="results/repetitions/regression/run_${i}"
    mkdir -p "$run_dir"

    files=(resultados_reg_*.txt)

    if [ ! -e "${files[0]}" ]; then
        echo "ERROR: run $i did not generate regression result files" >&2
        exit 1
    fi

    mv resultados_reg_*.txt "$run_dir/"

    echo "Run $i completed and stored in $run_dir"
done

echo "All regression runs completed."
