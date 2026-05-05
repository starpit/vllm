/*
 * ICB test with CORRECT storage mode (Shared instead of Private)
 * 
 * The crash was caused by using MTLResourceStorageModePrivate which doesn't
 * allow CPU access. We need MTLResourceStorageModeShared to encode commands
 * from the CPU.
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_storage_mode_fix.m -o test_icb_storage_mode_fix
 * Run: ./test_icb_storage_mode_fix
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
        printf("=== ICB Test with CORRECT Storage Mode ===\n\n");
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("Device: %s\n\n", [device.name UTF8String]);
        
        // Create pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shaderSource]
                                                      options:nil
                                                        error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"simple_write"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        
        // Create buffer
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:4 * sizeof(uint32_t) 
                                                         options:MTLResourceStorageModeShared];
        memset(outputBuffer.contents, 0, 4 * sizeof(uint32_t));
        
        // Test 1: inheritPipelineState=NO with SHARED storage mode
        printf("=== TEST 1: inheritPipelineState=NO, StorageMode=SHARED ===\n");
        MTLIndirectCommandBufferDescriptor* icbDesc1 = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc1.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc1.inheritPipelineState = NO;
        icbDesc1.inheritBuffers = NO;
        icbDesc1.maxKernelBufferBindCount = 1;
        
        // KEY FIX: Use MTLResourceStorageModeShared instead of Private
        id<MTLIndirectCommandBuffer> icb1 = [device newIndirectCommandBufferWithDescriptor:icbDesc1
                                                                           maxCommandCount:1
                                                                                   options:MTLResourceStorageModeShared];
        printf("✅ ICB created with SHARED storage mode\n");
        
        id<MTLIndirectComputeCommand> cmd1 = [icb1 indirectComputeCommandAtIndex:0];
        printf("✅ Got command at index 0 (no crash!)\n");
        
        [cmd1 reset];
        printf("✅ Reset command\n");
        
        printf("⚠️  Setting pipeline state...\n");
        [cmd1 setComputePipelineState:pipeline];
        printf("✅ Set pipeline state (NO CRASH!)\n");
        
        [cmd1 setKernelBuffer:outputBuffer offset:0 atIndex:0];
        printf("✅ Set buffer\n");
        
        [cmd1 concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ Set dispatch\n");
        
        // Execute
        printf("\n=== Executing ICB ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder executeCommandsInBuffer:icb1 withRange:NSMakeRange(0, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            printf("❌ Command buffer error: %s\n", [[commandBuffer.error localizedDescription] UTF8String]);
        } else {
            printf("✅ Command buffer completed\n");
        }
        
        uint32_t* output = (uint32_t*)outputBuffer.contents;
        printf("\nOutput: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
        
        if (output[0] == 42) {
            printf("\n🎉 TEST 1 SUCCESS: ICB executed correctly!\n");
            printf("The fix was using MTLResourceStorageModeShared instead of Private.\n");
            return 0;
        } else {
            printf("\n❌ TEST 1 FAILED: ICB did not execute\n");
            
            // Test 2: Try inheritPipelineState=YES as fallback
            printf("\n=== TEST 2: inheritPipelineState=YES, StorageMode=SHARED ===\n");
            memset(outputBuffer.contents, 0, 4 * sizeof(uint32_t));
            
            MTLIndirectCommandBufferDescriptor* icbDesc2 = [[MTLIndirectCommandBufferDescriptor alloc] init];
            icbDesc2.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
            icbDesc2.inheritPipelineState = YES;
            icbDesc2.inheritBuffers = NO;
            icbDesc2.maxKernelBufferBindCount = 1;
            
            id<MTLIndirectCommandBuffer> icb2 = [device newIndirectCommandBufferWithDescriptor:icbDesc2
                                                                               maxCommandCount:1
                                                                                       options:MTLResourceStorageModeShared];
            
            id<MTLIndirectComputeCommand> cmd2 = [icb2 indirectComputeCommandAtIndex:0];
            [cmd2 reset];
            [cmd2 setKernelBuffer:outputBuffer offset:0 atIndex:0];
            [cmd2 concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
            
            id<MTLCommandBuffer> commandBuffer2 = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder2 = [commandBuffer2 computeCommandEncoder];
            [encoder2 setComputePipelineState:pipeline];
            [encoder2 executeCommandsInBuffer:icb2 withRange:NSMakeRange(0, 1)];
            [encoder2 endEncoding];
            
            [commandBuffer2 commit];
            [commandBuffer2 waitUntilCompleted];
            
            output = (uint32_t*)outputBuffer.contents;
            printf("Output: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
            
            if (output[0] == 42) {
                printf("\n🎉 TEST 2 SUCCESS: ICB with inheritPipelineState=YES works!\n");
                return 0;
            } else {
                printf("\n❌ Both tests failed - compute ICBs may not be supported\n");
                return 1;
            }
        }
    }
    return 0;
}
