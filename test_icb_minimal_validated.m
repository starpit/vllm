/*
 * Minimal ICB test with GPU validation enabled
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_minimal_validated.m -o test_icb_minimal_validated
 * Run with validation: MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./test_icb_minimal_validated
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
        printf("=== ICB Test with Metal Debug Layer ===\n");
        printf("Run with: MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1\n\n");
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("Device: %s\n\n", [device.name UTF8String]);
        
        // Create pipeline
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shaderSource]
                                                      options:nil
                                                        error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"simple_write"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        printf("✅ Pipeline created\n");
        
        // Create buffer
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:4 * sizeof(uint32_t) 
                                                         options:MTLResourceStorageModeShared];
        memset(outputBuffer.contents, 0, 4 * sizeof(uint32_t));
        printf("✅ Buffer created\n\n");
        
        // Test: ICB with inheritPipelineState=NO (expected to crash)
        printf("=== TEST: inheritPipelineState=NO ===\n");
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = NO;
        icbDesc.inheritBuffers = NO;
        icbDesc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                           maxCommandCount:1
                                                                                   options:MTLResourceStorageModePrivate];
        printf("✅ ICB created\n");
        
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        printf("✅ Got command at index 0\n");
        
        [cmd reset];
        printf("✅ Reset command\n");
        
        printf("⚠️  About to call setComputePipelineState (expected crash point)...\n");
        fflush(stdout);
        
        [cmd setComputePipelineState:pipeline];
        
        printf("✅ Set pipeline state (NO CRASH!)\n");
        
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        printf("✅ Set buffer\n");
        
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ Set dispatch\n");
        
        // Execute
        printf("\n=== Executing ICB ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
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
            printf("✅ SUCCESS: ICB executed!\n");
        } else {
            printf("❌ FAILURE: ICB did not execute\n");
        }
    }
    return 0;
}