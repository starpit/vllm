/*
 * Minimal ICB test - incrementally add operations to find crash point
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_minimal.m -o test_icb_minimal
 * Run: ./test_icb_minimal
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
        
        // Create ICB descriptor
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = NO;
        icbDesc.inheritBuffers = NO;
        icbDesc.maxKernelBufferBindCount = 1;
        
        // Create ICB
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                          maxCommandCount:1
                                                                                  options:MTLResourceStorageModePrivate];
        printf("✅ Step 1: ICB created successfully\n");
        
        // Get command at index 0
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        printf("✅ Step 2: Got ICB command at index 0\n");
        
        // Reset command
        [cmd reset];
        printf("✅ Step 3: Reset command\n");
        
        // Try to set pipeline state - THIS IS WHERE IT CRASHES
        printf("⚠️  Step 4: About to call setComputePipelineState...\n");
        fflush(stdout);
        [cmd setComputePipelineState:pipeline];
        printf("✅ Step 4: Set pipeline state (if you see this, it didn't crash!)\n");
        
        // Set buffer
        printf("⚠️  Step 5: About to set buffer...\n");
        fflush(stdout);
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        printf("✅ Step 5: Set buffer\n");
        
        // Set dispatch
        printf("⚠️  Step 6: About to set dispatch...\n");
        fflush(stdout);
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ Step 6: Set dispatch\n");
        
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
            printf("✅ Command buffer completed successfully\n");
        }
        
        // Check results
        uint32_t* output = (uint32_t*)outputBuffer.contents;
        printf("\nOutput: [%u, %u, %u, %u]\n", output[0], output[1], output[2], output[3]);
        
        if (output[0] == 42) {
            printf("✅ SUCCESS!\n");
        } else {
            printf("❌ FAILURE: ICB did not execute\n");
        }
    }
    return 0;
}
