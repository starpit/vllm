/*
 * ICB test with inheritPipelineState=YES and SHARED storage mode
 * This is the configuration our Rust implementation uses
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_inherit_shared.m -o test_icb_inherit_shared
 * Run: ./test_icb_inherit_shared
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
        printf("=== ICB Test: inheritPipelineState=YES + SHARED storage ===\n\n");
        
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
        icbDesc.inheritPipelineState = YES;  // Pipeline set on encoder
        icbDesc.inheritBuffers = NO;
        icbDesc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                           maxCommandCount:1
                                                                                   options:MTLResourceStorageModeShared];
        printf("✅ ICB created (inheritPipelineState=YES, StorageMode=SHARED)\n");
        
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
        printf("✅ Got command at index 0\n");
        
        [cmd reset];
        printf("✅ Reset command\n");
        
        // Do NOT set pipeline on command (inherited from encoder)
        printf("⚠️  Setting buffer (no pipeline on command)...\n");
        [cmd setKernelBuffer:outputBuffer offset:0 atIndex:0];
        printf("✅ Set buffer\n");
        
        printf("⚠️  Setting dispatch...\n");
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        printf("✅ Set dispatch\n");
        
        // Execute with pipeline set on encoder
        printf("\n=== Executing ICB ===\n");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        printf("⚠️  Setting pipeline on encoder...\n");
        [encoder setComputePipelineState:pipeline];
        printf("✅ Pipeline set on encoder\n");
        
        printf("⚠️  Executing ICB commands...\n");
        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        printf("✅ ICB execution command encoded\n");
        
        [encoder endEncoding];
        printf("✅ Encoder ended\n");
        
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
            printf("\n🎉 SUCCESS: ICB executed correctly!\n");
            printf("Configuration that works:\n");
            printf("  - inheritPipelineState = YES\n");
            printf("  - StorageMode = SHARED\n");
            printf("  - Pipeline set on encoder, not ICB command\n");
            return 0;
        } else {
            printf("\n❌ FAILURE: ICB did not execute (output is zeros)\n");
            return 1;
        }
    }
    return 0;
}
