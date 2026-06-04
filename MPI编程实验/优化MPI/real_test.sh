#!/bin/bash
# Helper for the MPI workflow in qsub_mpi.sh.
# Official submit command from the PDF:
#   qsub qsub_mpi.sh

set -e

usage() {
    echo "Usage: bash real_test.sh [-O <opt>] [-s <seed>] [-n <np>]"
    echo "  -O, --opt   optional: O0/O1/O2/O3/Ofast or 0/1/2/3/fast, default O2"
    echo "  -s, --seed  optional: random seed passed to main through SVD_SEED"
    echo "  -n, --np    optional: MPI process count, must be <= nodes * ppn in qsub_mpi.sh"
    echo ""
    echo "Edit qsub_mpi.sh to set nodes and ppn, then submit with qsub qsub_mpi.sh."
}

normalize_opt() {
    local in="$1"
    case "$in" in
        O0|O1|O2|O3|Ofast) echo "-$in" ;;
        0|1|2|3) echo "-O$in" ;;
        fast) echo "-Ofast" ;;
        *) return 1 ;;
    esac
}

OPT_INPUT="O2"
SEED=""
NP=""

while [ $# -gt 0 ]; do
    case "$1" in
        -O|--opt)
            OPT_INPUT="${2:-}"
            shift 2
            ;;
        -s|--seed)
            SEED="${2:-}"
            shift 2
            ;;
        -n|--np)
            NP="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if ! OPT_FLAG="$(normalize_opt "$OPT_INPUT")"; then
    echo "Unsupported optimization level: $OPT_INPUT"
    usage
    exit 1
fi

if [ -n "$NP" ]; then
    case "$NP" in
        *[!0-9]*)
            echo "np must be a positive integer"
            usage
            exit 1
            ;;
    esac

    if [ "$NP" -lt 1 ] || [ "$NP" -gt 32 ]; then
        echo "np must be between 1 and 32"
        usage
        exit 1
    fi
fi

MPICXX=${MPICXX:-mpic++}

"$MPICXX" main.cpp gkh.cpp bidiagonalization.cpp -o main \
    "$OPT_FLAG" -DSVD_ENABLE_MPI -fopenmp -lpthread -std=c++17

QSUB_VARS=""
if [ -n "$SEED" ]; then
    QSUB_VARS="SVD_SEED=${SEED}"
fi
if [ -n "$NP" ]; then
    if [ -n "$QSUB_VARS" ]; then
        QSUB_VARS="${QSUB_VARS},SVD_MPI_NP=${NP}"
    else
        QSUB_VARS="SVD_MPI_NP=${NP}"
    fi
fi
if [ -n "$QSUB_VARS" ]; then
    qsub -v "$QSUB_VARS" qsub_mpi.sh
else
    qsub qsub_mpi.sh
fi
