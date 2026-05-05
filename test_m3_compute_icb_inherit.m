/*
 * Test for Metal Compute ICBs with inheritPipelineState=YES on M3
 * 
 * Compile: clang -framework Metal -framework Foundation test_m3_compute_icb_inherit.m -o test_m3_icb_inherit
 * Run: ./test_m3_icb_inherit
 * 
 * This tests the configuration we used in our Rust implementation:
 * - inheritPipelineState = YES (pipeline set on encoder, not ICB command)
 * - inheritBuffers = NO (buffers set on ICB command)
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

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
        
        // Create output buffer
        const NSUInteger bufferSize = 4 * sizeof(uint32_t);
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        memset(outputBuffer.contents, 0, bufferSize);
        
        // Create ICB descriptor - INHERIT PIPELINE STATE
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = YES;  // Pipeline set on encoder
        icbDesc.inheritBuffers = NO;         // Buffers set on ICB command
        icbDesc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                          maxCommandCount:1
                                                                                  options:MTLResourceStorageModePrivate];
        if (!icb) {
            printf("ERROR: Failed to create ICB\n");
            return 1;
        }
        
        printf("\n=== Configuration: inheritPipelineState=YES ===\n");
        printf("Pipeline will be set on encoder, not ICB command\n");
        
        // Encode command into ICB
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        [cmd reset];
        // DO NOT set pipeline on command (it's inherited from encoder)
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        
        printf("ICB command encoded (no pipeline set on command)\n");
        
        // Execute ICB
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline on ENCODER (inherited by ICB commands)
        [encoder setComputePipelineState:pipeline];
        printf("Pipeline set on encoder: %p\n", (void*)pipeline);
        
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
            printf("\n✅ SUCCESS: ICB with inheritPipelineState=YES works!\n");
            printf("This is the configuration used in our Rust implementation.\n");
            return 0;
        } else {
            printf("\n❌ FAILURE: ICB with inheritPipelineState=YES did not execute.\n");
            
            // Baseline test
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
            
            return 1;
        }
    }
    return 0;
}
