#!/bin/bash
# Usage: bash test.sh [LAB] [NODES] [CORES] [-O <opt>] [-s <seed>]

usage() {
    echo "正确格式：bash test.sh [LAB] [NODES] [CORES] [-O <opt>] [-s <seed>]"
    echo "  -O, --opt   可选，支持 O0/O1/O2/O3/Ofast 或 0/1/2/3/fast，默认 O2"
    echo "  -s, --seed  可选，矩阵初始化种子，默认 20260409"
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

if [ $# -lt 3 ]; then
    echo "参数缺失"
    usage
    exit 1
fi

ID=$USER
LAB="${1}"        # 实验编号
NODES="${2}"      # 申请节点数
CORES="${3}"      # 每个节点申请的核心数
shift 3

OPT_INPUT="O2"
SEED="20260409"

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
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

if ! OPT_FLAG="$(normalize_opt "$OPT_INPUT")"; then
    echo "不支持的优化等级: $OPT_INPUT"
    usage
    exit 1
fi

LOG_PATH="/parallel_hw/svd/${LAB}/"
RESULT_PATH="/home/${ID}/svd/"

# 参数校验
if [ "$NODES" -gt 4 ]; then
    echo "计算节点申请过多（最大 4）"
    exit 1
fi

if [ "$CORES" -lt 1 ] || [ "$CORES" -gt 8 ]; then
    echo "核心数无效（1-8）"
    exit 1
fi

# 清理旧文件
rm -f "${RESULT_PATH}test.o" "${RESULT_PATH}test.e"

MPICXX=${MPICXX:-mpic++}
NP=$((NODES * CORES))

"$MPICXX" main.cpp gkh.cpp bidiagonalization.cpp -o main \
    "$OPT_FLAG" -DSVD_ENABLE_MPI -fopenmp -lpthread -std=c++17

jobid=$(qsub -v SVD_SEED="$SEED",SVD_MPI_NP="$NP" -l nodes="${NODES}:ppn=${CORES}" qsub_mpi.sh)

echo "Submitted job with ID: $jobid"
echo "Compile opt: $OPT_FLAG"
echo "Seed: $SEED"
echo "MPI np: $NP"

timeout=600
elapsed=0
interval=5
timed_out=false

while [ ! -f "${RESULT_PATH}test.o" ] && [ $elapsed -lt $timeout ]; do
    sleep $interval
    elapsed=$((elapsed + interval))
done

# 超时
if [ ! -f "${RESULT_PATH}test.o" ]; then
    timed_out=true
    echo "运行时间超过10分钟，结果请查看 ${RESULT_PATH}test.e 和 ${RESULT_PATH}test.o"
    exit 1
fi

# 非超时
if [ "$timed_out" = false ]; then
    if [ -f "${RESULT_PATH}test.o" ]; then
        output_o=$(cat "${RESULT_PATH}test.o")
        output_e=$(cat "${RESULT_PATH}test.e")
    else
        output_o="Output file ${RESULT_PATH}test.o not found."
        output_e="Error file ${RESULT_PATH}test.e not found"
    fi

    echo "${output_e}"
    echo "${output_o}"
fi

# 结构化提取日志摘要
current_time=$(date +"%Y-%m-%d-%H-%M-%S")
log_file="${LOG_PATH}${ID}_${LAB}.log"

# 提取随机种子、编译优化、每个样例通过情况、总通过情况
truncated_output=$(
    awk -v opt="$OPT_FLAG" '
    /^=== .* ===$/{
        sample_count++
        sample_name=$0
        gsub(/^=== /, "", sample_name)
        gsub(/ ===$/, "", sample_name)
        names[sample_count]=sample_name
        next
    }
    /结果:[[:space:]]*(PASS|FAIL)/{
        status=$0
        sub(/.*结果:[[:space:]]*/, "", status)
        status_by_idx[sample_count]=status
        next
    }
    /^随机种子基值:/{
        seed=$0
        next
    }
    /^通过:[[:space:]]*/{
        total=$0
        next
    }
    END{
        if (seed!="") {
            print seed
            print "编译优化: " opt
        } else {
            print "随机种子基值: N/A"
            print "编译优化: " opt
        }
        for (i=1; i<=sample_count; i++) {
            s=status_by_idx[i]
            if (s=="") s="UNKNOWN"
            print "样例" i "（" names[i] "）: " s
        }
        if (total!="") print total
    }' "${RESULT_PATH}test.o" 2>/dev/null
)

# 从第5个样例块中直接截取两行耗时（固定取块内第10、11行）
sample5_time_lines=$(
    awk '
    /^=== .* ===$/ {
        block++
        in_block=(block==5)
    }
    in_block {
        print
    }
    block>5 {
        exit
    }' "${RESULT_PATH}test.o" 2>/dev/null | sed -n '10,11p'
)

if [ -z "$sample5_time_lines" ]; then
    sample5_time_lines="N/A"
fi

truncated_output="${truncated_output}"$'\n'"样例5耗时:"$'\n'"${sample5_time_lines}"

if [ -z "$truncated_output" ]; then
    truncated_output="No structured output available"
fi
{
    echo "test time: $current_time"
    echo "$truncated_output"
    echo "----------------------------------------"
    echo ""
} >> "$log_file"
