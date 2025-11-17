#!/bin/bash

echo "Testing gemma3.py vs gemma3_packed.py"
echo "======================================"

# Test 1: CPU
echo -e "\nTest 1: CPU"
CUDA_VISIBLE_DEVICES="" python test_gemma3_vs_packed.py
T1=$?

# Test 2: GPU eager/sdpa
echo -e "\nTest 2: GPU eager/sdpa for packed"
ATTN_IMPLEMENTATION="eager" python test_gemma3_vs_packed.py
T2=$?

# Test 3: GPU Flash Attention 2
echo -e "\nTest 3: GPU (Flash Attention 2)"
ATTN_IMPLEMENTATION="flash_attention_2" python test_gemma3_vs_packed.py
T3=$?

# Summary
echo -e "\n======================================"
echo "Results:"
echo "  CPU:           $([ $T1 -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  GPU (eager/sdpa):   $([ $T2 -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  GPU (Flash):   $([ $T3 -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "======================================"

[ $T1 -eq 0 ] && [ $T2 -eq 0 ] && [ $T3 -eq 0 ]
