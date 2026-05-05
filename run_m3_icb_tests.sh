#!/bin/bash
# Script to compile and run M3 Compute ICB tests
# Run this on your M3 Mac to test if compute ICBs are supported

set -e

echo "=========================================="
echo "M3 Compute ICB Test Suite"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Compiling test programs..."
echo ""

# Compile test 1
echo "Compiling test_m3_compute_icb.m..."
clang -framework Metal -framework Foundation test_m3_compute_icb.m -o test_m3_icb
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile test_m3_compute_icb.m"
    exit 1
fi

# Compile test 2
echo "Compiling test_m3_compute_icb_inherit.m..."
clang -framework Metal -framework Foundation test_m3_compute_icb_inherit.m -o test_m3_icb_inherit
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile test_m3_compute_icb_inherit.m"
    exit 1
fi

echo ""
echo "=========================================="
echo "TEST 1: Standard ICB (inheritPipelineState=NO)"
echo "=========================================="
echo ""
./test_m3_icb
TEST1_RESULT=$?

echo ""
echo "=========================================="
echo "TEST 2: Inherited Pipeline (inheritPipelineState=YES)"
echo "=========================================="
echo ""
./test_m3_icb_inherit
TEST2_RESULT=$?

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Test 1 (inheritPipelineState=NO): $([ $TEST1_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo "Test 2 (inheritPipelineState=YES): $([ $TEST2_RESULT -eq 0 ] && echo '✅ PASSED' || echo '❌ FAILED')"
echo ""

if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ]; then
    echo "🎉 Both tests PASSED!"
    echo "Compute ICBs ARE supported on this GPU (likely Apple9/M3)."
    echo "The M1 Max limitation is hardware-specific to Apple7 GPU family."
elif [ $TEST1_RESULT -ne 0 ] && [ $TEST2_RESULT -ne 0 ]; then
    echo "⚠️  Both tests FAILED!"
    echo "Compute ICBs are NOT supported on this GPU."
    echo "Recommendation: Use direct encoder recording for Phase 4.6."
elif [ $TEST1_RESULT -eq 0 ]; then
    echo "✅ Test 1 PASSED, Test 2 FAILED"
    echo "Only inheritPipelineState=NO works."
    echo "Recommendation: Update Rust implementation to set pipeline on ICB commands."
else
    echo "✅ Test 2 PASSED, Test 1 FAILED"
    echo "Only inheritPipelineState=YES works."
    echo "Our Rust implementation approach is correct!"
    echo "Issue may be specific to M1 Max (Apple7) hardware."
fi

echo ""
echo "Please share these results with the development team."
echo ""

# Clean up binaries
rm -f test_m3_icb test_m3_icb_inherit

exit 0
