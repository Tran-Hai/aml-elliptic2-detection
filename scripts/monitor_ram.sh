#!/bin/bash
# Monitor RAM usage during processing

echo "Monitoring RAM usage..."
echo "Press Ctrl+C to stop"
echo ""

LOG_FILE="logs/ram_usage.log"
mkdir -p logs

echo "timestamp,total_ram,available_ram,used_ram,percent_used,process_ram" > "$LOG_FILE"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # System RAM info
    ram_info=$(free -m | awk 'NR==2{printf "%d,%d,%d,%.2f", $2, $7, $3, $3*100/$2}')
    
    # Process RAM (if python script is running)
    process_ram=$(ps aux | grep python | grep -v grep | awk '{sum+=$6} END {print sum/1024}')
    if [ -z "$process_ram" ]; then
        process_ram="0"
    fi
    
    echo "$timestamp,$ram_info,$process_ram" >> "$LOG_FILE"
    
    # Display current status
    clear
    echo "========================================"
    echo "RAM Usage Monitor"
    echo "========================================"
    echo "Timestamp: $timestamp"
    free -h
    echo ""
    echo "Process RAM: ${process_ram} MB"
    echo ""
    echo "Log file: $LOG_FILE"
    echo "Press Ctrl+C to stop"
    
    sleep 5
done
