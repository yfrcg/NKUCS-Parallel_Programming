#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=4:ppn=8

# MPI script parameters:
# - nodes/ppn are set in the #PBS -l line above.
# - NP is the number of MPI processes to launch.
# Keep: NP <= nodes * ppn, nodes <= 4, ppn <= 8.
NP=${SVD_MPI_NP:-8}
PROJECT=svd
MASTER=master_ubss1
PROJECT_DIR="/home/${USER}/${PROJECT}"
WORK_DIR="/home/${USER}"

NODE_COUNT=$(sort -u "$PBS_NODEFILE" | wc -l)
SLOT_COUNT=$(wc -l < "$PBS_NODEFILE")
PPN=$((SLOT_COUNT / NODE_COUNT))

case "$NP" in
    ''|*[!0-9]*)
        echo "np must be a positive integer, got ${NP}" >&2
        exit 1
        ;;
esac

if [ "$NODE_COUNT" -gt 4 ]; then
    echo "nodes must be <= 4, got ${NODE_COUNT}" >&2
    exit 1
fi

if [ "$PPN" -gt 8 ]; then
    echo "ppn must be <= 8, got ${PPN}" >&2
    exit 1
fi

if [ "$NP" -lt 1 ] || [ "$NP" -gt "$SLOT_COUNT" ]; then
    echo "np must satisfy 1 <= np <= nodes * ppn (${SLOT_COUNT}), got ${NP}" >&2
    exit 1
fi

NODES=$(sort -u "$PBS_NODEFILE")

for node in $NODES; do
    ssh "$node" "mkdir -p ${WORK_DIR}" 1>&2
    scp "${MASTER}:${PROJECT_DIR}/main" "${node}:${WORK_DIR}/" 1>&2
    scp -r "${MASTER}:${PROJECT_DIR}/files" "${node}:${WORK_DIR}/" 1>&2
done

if [ -n "$SVD_SEED" ]; then
    /usr/local/bin/mpiexec -np "$NP" -machinefile "$PBS_NODEFILE" \
        env SVD_MPI_RELAUNCHED=1 SVD_WORKERS="$NP" /home/"${USER}"/main "$SVD_SEED"
else
    /usr/local/bin/mpiexec -np "$NP" -machinefile "$PBS_NODEFILE" \
        env SVD_MPI_RELAUNCHED=1 SVD_WORKERS="$NP" /home/"${USER}"/main
fi

scp -r "${WORK_DIR}/files/" "${MASTER}:${PROJECT_DIR}/" 2>&1
