/*
 * Minimal test for Metal Compute Indirect Command Buffers on M3 (Apple9 GPU)
 * 
 * Compile: clang -framework Metal -framework Foundation test_m3_compute_icb.m -o test_m3_icb
 * Run: ./test_m3_icb
 * 
 * This test verifies if compute ICBs (ConcurrentDispatch) work on Apple9 GPU family.
 * Expected output if ICB works: "SUCCESS: ICB executed correctly! Output: 42"
 * Expected output if ICB fails: "FAILURE: ICB did not execute. Output: 0"
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Simple compute shader that writes a constant to output buffer
static const char* shaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void simple_write(device uint* output [[buffer(0)]],
                        uint gid [[thread_position_in_grid]])
{
    output[gid] = 42;
}
)";

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device found\n");
            return 1;
        }
        
        printf("Testing on: %s\n", [device.name UTF8String]);
        printf("GPU Family: Apple%lu\n", (unsigned long)[device supportsFamily:MTLGPUFamilyApple9] ? 9 : 
                                         [device supportsFamily:MTLGPUFamilyApple8] ? 8 :
                                         [device supportsFamily:MTLGPUFamilyApple7] ? 7 : 0);
        
        // Create compute pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shaderSource]
                                                      options:nil
                                                        error:&error];
        if (!library) {
            printf("ERROR: Failed to compile shader: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        
        id<MTLFunction> function = [library newFunctionWithName:@"simple_write"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            printf("ERROR: Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        
        // Create output buffer (4 elements)
        const NSUInteger bufferSize = 4 * sizeof(uint32_t);
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        memset(outputBuffer.contents, 0, bufferSize);
        
        // Create ICB descriptor for compute commands
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = NO;  // We'll set pipeline on the command
        icbDesc.inheritBuffers = NO;        // We'll set buffers on the command
        icbDesc.maxKernelBufferBindCount = 1;
        
        // Create ICB
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                          maxCommandCount:1
                                                                                  options:MTLResourceStorageModePrivate];
        if (!icb) {
            printf("ERROR: Failed to create ICB\n");
            return 1;
        }
        
        printf("\n=== Encoding ICB command from CPU ===\n");
        
        // Encode command into ICB (from CPU)
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        [cmd reset];
        [cmd setComputePipelineState:pipeline];
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        
        printf("ICB command encoded:\n");
        printf("  - Pipeline: %p\n", (void*)pipeline);
        printf("  - Buffer: %p\n", (void*)outputBuffer);
        printf("  - Dispatch: 1 threadgroup, 4 threads\n");
        
        // Create command queue and buffer
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        // Execute ICB on compute encoder
        printf("\n=== Executing ICB ===\n");
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Note: When inheritPipelineState=NO, we should NOT set pipeline on encoder
        // The ICB command has its own pipeline state
        
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Check results
        printf("\n=== Results ===\n");
        uint32_t* output = (uint32_t*)outputBuffer.contents;
        printf("Output buffer: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
        
        BOOL success = YES;
        for (int i = 0; i < 4; i++) {
            if (output[i] != 42) {
                success = NO;
                break;
            }
        }
        
        if (success) {
            printf("\n✅ SUCCESS: ICB executed correctly!\n");
            printf("Compute ICBs (ConcurrentDispatch) ARE supported on this GPU.\n");
            return 0;
        } else {
            printf("\n❌ FAILURE: ICB did not execute.\n");
            printf("Compute ICBs (ConcurrentDispatch) are NOT supported on this GPU.\n");
            
            // Try direct dispatch as baseline
            printf("\n=== Testing direct dispatch (baseline) ===\n");
            memset(outputBuffer.contents, 0, bufferSize);
            
            id<MTLCommandBuffer> directCmdBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> directEncoder = [directCmdBuffer computeCommandEncoder];
            [directEncoder setComputePipelineState:pipeline];
            [directEncoder setBuffer:outputBuffer offset:0 atIndex:0];
            [directEncoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
            [directEncoder endEncoding];
            [directCmdBuffer commit];
            [directCmdBuffer waitUntilCompleted];
            
            output = (uint32_t*)outputBuffer.contents;
            printf("Direct dispatch output: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
            
            BOOL directSuccess = YES;
            for (int i = 0; i < 4; i++) {
                if (output[i] != 42) {
                    directSuccess = NO;
                    break;
                }
            }
            
            if (directSuccess) {
                printf("✅ Direct dispatch works (GPU and shader are functional)\n");
                printf("This confirms the issue is specifically with compute ICBs.\n");
            } else {
                printf("❌ Direct dispatch also failed (GPU/shader issue)\n");
            }
            
            return 1;
        }
    }
    return 0;
}
