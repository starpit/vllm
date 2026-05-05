/*
 * Check if device supports ICB feature sets
 * 
 * Compile: clang -framework Metal -framework Foundation test_icb_feature_check.m -o test_icb_feature_check
 * Run: ./test_icb_feature_check
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("ERROR: No Metal device found\n");
            return 1;
        }
        
        printf("Device: %s\n", [device.name UTF8String]);
        printf("\n=== GPU Family Support ===\n");
        printf("Apple7: %s\n", [device supportsFamily:MTLGPUFamilyApple7] ? "YES" : "NO");
        printf("Apple8: %s\n", [device supportsFamily:MTLGPUFamilyApple8] ? "YES" : "NO");
        printf("Apple9: %s\n", [device supportsFamily:MTLGPUFamilyApple9] ? "YES" : "NO");
        
        // Try to create an ICB to verify API support
        printf("\n=== Attempting ICB Creation ===\n");
        MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        desc.inheritPipelineState = YES;
        desc.inheritBuffers = NO;
        desc.maxKernelBufferBindCount = 1;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:desc
                                                                          maxCommandCount:1
                                                                                  options:MTLResourceStorageModePrivate];
        
        if (icb) {
            printf("✅ ICB creation succeeded\n");
            printf("   ICB size: %llu bytes\n", (unsigned long long)[icb size]);
            printf("\n=== Conclusion ===\n");
            printf("Device supports ICB API for compute commands.\n");
            printf("However, our tests show compute ICBs do NOT execute on Apple Silicon.\n");
            printf("This is likely a Metal limitation, not a device capability issue.\n");
        } else {
            printf("❌ ICB creation FAILED\n");
            printf("\n=== Conclusion ===\n");
            printf("Device does not support ICB at the API level.\n");
        }
    }
    return 0;
}