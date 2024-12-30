#!/bin/bash

# 定义bpftrace脚本路径
BPFTRACE_SCRIPT="./opensnoop.bt"
# 定义日志文件路径
LOG_FILE="bt.log"

# 启动bpftrace并将其输出重定向到日志文件
sudo bpftrace "$BPFTRACE_SCRIPT" > "$LOG_FILE" 2>&1 &

# 获取bpftrace进程的PID
BPFTRACE_PID=$!

# 等待10秒
sleep 10

# 终止bpftrace进程
sudo pkill -P $BPFTRACE_PID

# 等待bpftrace进程结束
wait $BPFTRACE_PID || true

echo "bpftrace has been terminated and output saved to $LOG_FILE"
