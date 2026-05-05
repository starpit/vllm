/*
 * Test ICB with inheritPipelineState=YES (DON'T call setComputePipelineState)
 * 
 * Maybe the solution is to NEVER call setComputePipelineState on the ICB command.
 * Instead, set it on the encoder and let the ICB inherit it.
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_inherit_only.m -o test_icb_inherit_only
 * Run: ./test_icb_inherit_only
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
        printf("=== ICB Test: inheritPipelineState=YES (NEVER call setComputePipelineState) ===\n\n");
        
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
        
        // Create ICB with inheritPipelineState=YES and SHARED storage
        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = YES;  // Pipeline comes from encoder
        icbDesc.inheritBuffers = NO;         // We'll set buffers on ICB
        icbDesc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                           maxCommandCount:1
                                                                                   options:MTLResourceStorageModeShared];
        printf("✅ ICB created (inheritPipelineState=YES, StorageMode=SHARED)\n");
        
        // Encode ICB command - DO NOT call setComputePipelineState
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        printf("✅ Got command at index 0\n");
        
        [cmd reset];
        printf("✅ Reset command\n");
        
        // SKIP setComputePipelineState - that's what causes the crash!
        printf("⚠️  Skipping setComputePipelineState (will inherit from encoder)\n");
        
        // Only set buffer and dispatch
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        printf("✅ Set buffer\n");
        
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ Set dispatch\n");
        
        // Execute - set pipeline on ENCODER, not ICB
        printf("\n=== Executing ICB ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline on encoder - ICB will inherit it
        [encoder setComputePipelineState:pipeline];
        printf("✅ Pipeline set on ENCODER (ICB will inherit)\n");
        
        // Execute ICB
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
            printf("\n🎉🎉🎉 SUCCESS! ICB EXECUTED CORRECTLY! 🎉🎉🎉\n");
            printf("\nThe solution:\n");
            printf("1. Use inheritPipelineState=YES\n");
            printf("2. Use MTLResourceStorageModeShared (not Private)\n");
            printf("3. NEVER call setComputePipelineState on ICB command\n");
            printf("4. Set pipeline on encoder before executeCommandsInBuffer\n");
            printf("5. ICB command only sets buffers and dispatch parameters\n");
            return 0;
        } else {
            printf("\n❌ FAILURE: ICB did not execute (output is zeros)\n");
            return 1;
        }
    }
    return 0;
}
