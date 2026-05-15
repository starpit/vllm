/*
 * ICB test with pipeline that SUPPORTS indirect command buffers
 * 
 * The Metal debug layer revealed the issue:
 * "compute pipeline set on this encoder does not support indirect command buffers"
 * 
 * Solution: Set supportIndirectCommandBuffers = YES on the pipeline descriptor!
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_pipeline_support.m -o test_icb_pipeline_support
 * Run: ./test_icb_pipeline_support
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
        printf("=== ICB Test with Pipeline Support for ICB ===\n\n");
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("Device: %s\n\n", [device.name UTF8String]);
        
        // Create library and function
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shaderSource]
                                                      options:nil
                                                        error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"simple_write"];
        
        // Create pipeline descriptor with ICB support enabled
        MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
        pipelineDesc.computeFunction = function;
        pipelineDesc.supportIndirectCommandBuffers = YES;  // THIS IS THE KEY!
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pipelineDesc
                                                                                      options:0
                                                                                   reflection:nil
                                                                                        error:&error];
        if (!pipeline) {
            printf("❌ Failed to create pipeline: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        printf("✅ Pipeline created with supportIndirectCommandBuffers=YES\n");
        
        // Create buffer
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:4 * sizeof(uint32_t) 
                                                         options:MTLResourceStorageModeShared];
        memset(outputBuffer.contents, 0, 4 * sizeof(uint32_t));
        
        // Create ICB with inheritPipelineState=YES and SHARED storage
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = YES;
        icbDesc.inheritBuffers = NO;
        icbDesc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                           maxCommandCount:1
                                                                                   options:MTLResourceStorageModeShared];
        printf("✅ ICB created (inheritPipelineState=YES, StorageMode=SHARED)\n");
        
        // Encode ICB command
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        [cmd reset];
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ ICB command encoded (buffer + dispatch, no pipeline)\n");
        
        // Execute
        printf("\n=== Executing ICB ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        printf("✅ Pipeline (with ICB support) set on encoder\n");
        
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        printf("✅ ICB execution command encoded\n");
        
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            printf("❌ Command buffer error: %s\n", [[commandBuffer.error localizedDescription] UTF8String]);
        } else {
            printf("✅ Command buffer completed successfully\n");
        }
        
        // Check results
        uint32_t* output = (uint32_t*)outputBuffer.contents;
        printf("\nOutput: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
        
        if (output[0] == 42 && output[1] == 42 && output[2] == 42 && output[3] == 42) {
            printf("\n🎉🎉🎉 SUCCESS! COMPUTE ICBs WORK ON APPLE SILICON! 🎉🎉🎉\n");
            printf("\nThe solution:\n");
            printf("1. Create pipeline with supportIndirectCommandBuffers=YES\n");
            printf("2. Use inheritPipelineState=YES on ICB descriptor\n");
            printf("3. Use MTLResourceStorageModeShared for ICB\n");
            printf("4. Set pipeline on encoder before executeCommandsInBuffer\n");
            printf("5. ICB command only sets buffers and dispatch parameters\n");
            printf("\nCompute ICBs ARE functional on Apple Silicon when configured correctly!\n");
            return 0;
        } else {
            printf("\n❌ FAILURE: ICB did not execute\n");
            return 1;
        }
    }
    return 0;
}
