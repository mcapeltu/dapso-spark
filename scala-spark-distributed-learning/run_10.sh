mkdir -p results/repetitions/classification

for i in $(seq -w 1 10); do
    rm -f resultados_*.txt
    sbt run
    mkdir -p "results/repetitions/classification/run_$i"
    mv resultados_iters_*.txt \
       "results/repetitions/classification/run_$i/"
done
