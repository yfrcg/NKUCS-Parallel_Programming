#!/bin/sh
#PBS -N qsub
#PBS -e test.e
#PBS -o test.o

SOURCE_DIR="${SVD_WORKDIR:-${PBS_O_WORKDIR:-/home/${USER}/svd/OpenMP}}"
JOB_TAG="${PBS_JOBID%%.*}"
RUN_MAIN="/home/${USER}/main_openmp_${JOB_TAG}"

scp "master_ubss1:${SOURCE_DIR}/main" "$RUN_MAIN" 1>&2 || exit 1
chmod 700 "$RUN_MAIN" || exit 1

if [ -n "${SVD_SEED:-}" ]; then
    "$RUN_MAIN" "$SVD_SEED"
else
    "$RUN_MAIN"
fi
status=$?

rm -f "$RUN_MAIN"
exit $status
