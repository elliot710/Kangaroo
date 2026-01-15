/*
 * Metal Engine Implementation for Pollard's Kangaroo Algorithm
 * Based on JeanLucPons/Kangaroo CUDA implementation
 *
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 *
 * Optimized for Apple Silicon (M1/M2/M3) GPUs
 */

#ifdef __APPLE__

// Prevent MacTypes.h from defining its own Point struct
#define Point MacPoint
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#undef Point

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>  // for std::min
#include <signal.h>
#include <atomic>
#include <sched.h>    // for sched_yield

// Now include our headers with our Point class
#include "MetalEngine.h"
#include "../Timer.h"
#include "../Kangaroo.h"

// Global shutdown flag for Metal operations - checked in Run loop
static std::atomic<bool> g_metalShutdown(false);

// ---------------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------------

MetalEngine::MetalEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound) {
    // Note: Signal handling is done in main.cpp
    
    this->nbThreadPerGroup = nbThreadPerGroup;
    this->nbThread = nbThreadGroup * nbThreadPerGroup;
    this->maxFound = maxFound;
    this->initialised = false;
    this->lostWarning = false;
    this->wildOffset.SetInt32(0);

    // Initialize pointers
    device = nullptr;
    commandQueue = nullptr;
    computePipeline = nullptr;
    initPipeline = nullptr;
    kangarooBuffer = nullptr;
    jumpDistBuffer = nullptr;
    jumpPxBuffer = nullptr;
    jumpPyBuffer = nullptr;
    foundCountBuffer = nullptr;
    outputBuffer = nullptr;
    dpMaskBuffer = nullptr;
    maxFoundBuffer = nullptr;
    kangarooPinned = nullptr;
    outputPinned = nullptr;
    jumpPinned = nullptr;

    if (!InitMetal(gpuId)) {
        printf("MetalEngine: Failed to initialize Metal\n");
        return;
    }

    if (!CreatePipelines()) {
        printf("MetalEngine: Failed to create pipelines\n");
        return;
    }

    if (!CreateBuffers()) {
        printf("MetalEngine: Failed to create buffers\n");
        return;
    }

    initialised = true;
}

// ---------------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------------

MetalEngine::~MetalEngine() {
    // Release Metal objects (ARC handles this in Objective-C)
    @autoreleasepool {
        if (kangarooBuffer) {
            CFRelease(kangarooBuffer);
        }
        if (jumpDistBuffer) {
            CFRelease(jumpDistBuffer);
        }
        if (jumpPxBuffer) {
            CFRelease(jumpPxBuffer);
        }
        if (jumpPyBuffer) {
            CFRelease(jumpPyBuffer);
        }
        if (foundCountBuffer) {
            CFRelease(foundCountBuffer);
        }
        if (outputBuffer) {
            CFRelease(outputBuffer);
        }
        if (dpMaskBuffer) {
            CFRelease(dpMaskBuffer);
        }
        if (maxFoundBuffer) {
            CFRelease(maxFoundBuffer);
        }
        if (commandQueue) {
            CFRelease(commandQueue);
        }
        if (computePipeline) {
            CFRelease(computePipeline);
        }
        if (initPipeline) {
            CFRelease(initPipeline);
        }
        if (device) {
            CFRelease(device);
        }
    }

    // Free host memory
    if (kangarooPinned) {
        free(kangarooPinned);
    }
    if (outputPinned) {
        free(outputPinned);
    }
    if (jumpPinned) {
        free(jumpPinned);
    }
}

// ---------------------------------------------------------------------------------
// Metal Initialization
// ---------------------------------------------------------------------------------

bool MetalEngine::InitMetal(int gpuId) {
    @autoreleasepool {
        // Get all Metal devices
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        
        if (devices.count == 0) {
            printf("MetalEngine: No Metal devices found\n");
            return false;
        }

        // Select device
        id<MTLDevice> mtlDevice = nil;
        if (gpuId >= 0 && gpuId < (int)devices.count) {
            mtlDevice = devices[gpuId];
        } else {
            mtlDevice = MTLCreateSystemDefaultDevice();
        }

        if (!mtlDevice) {
            printf("MetalEngine: Failed to get Metal device\n");
            return false;
        }

        device = (__bridge_retained void*)mtlDevice;

        // Get device info
        deviceName = std::string([mtlDevice.name UTF8String]);

        // Append GPU info
        char tmp[512];
        uint64_t maxThreadsPerThreadgroup = mtlDevice.maxThreadsPerThreadgroup.width;
        uint64_t recommendedMaxWorkingSetSize = mtlDevice.recommendedMaxWorkingSetSize / (1024 * 1024);
        
        snprintf(tmp, sizeof(tmp), "GPU #%d %s (Max %llu threads/group, %.1f MB recommended)",
                gpuId, deviceName.c_str(), 
                maxThreadsPerThreadgroup,
                (double)recommendedMaxWorkingSetSize);
        deviceName = std::string(tmp);

        // Create command queue
        id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
        if (!queue) {
            printf("MetalEngine: Failed to create command queue\n");
            return false;
        }
        commandQueue = (__bridge_retained void*)queue;

        printf("MetalEngine: Using %s\n", deviceName.c_str());
        return true;
    }
}

// ---------------------------------------------------------------------------------
// Create Compute Pipelines
// ---------------------------------------------------------------------------------

bool MetalEngine::CreatePipelines() {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        NSError* error = nil;

        // Find the Metal library
        NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"KangarooKernel" ofType:@"metallib"];
        
        id<MTLLibrary> library = nil;
        
        if (libraryPath) {
            library = [mtlDevice newLibraryWithFile:libraryPath error:&error];
        }
        
        // Try current directory for the precompiled library
        if (!library) {
            libraryPath = @"KangarooKernel.metallib";
            library = [mtlDevice newLibraryWithFile:libraryPath error:&error];
        }
        
        if (!library) {
            // Try to compile from source
            NSString* shaderPath = @"Metal/KangarooKernel.metal";
            NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                               encoding:NSUTF8StringEncoding
                                                                  error:&error];
            
            if (!shaderSource) {
                shaderPath = @"./KangarooKernel.metal";
                shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            }
            
            if (shaderSource) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#ifdef USE_SYMMETRY
                options.preprocessorMacros = @{@"USE_SYMMETRY": @1};
#endif
                library = [mtlDevice newLibraryWithSource:shaderSource options:options error:&error];
            }
            
            if (!library) {
                printf("MetalEngine: Failed to load shader library: %s\n",
                       error ? [[error localizedDescription] UTF8String] : "Unknown error");
                return false;
            }
        }

        // Create compute kernel
        id<MTLFunction> computeFunction = [library newFunctionWithName:@"computeKangaroos"];
        if (!computeFunction) {
            printf("MetalEngine: Failed to find computeKangaroos function\n");
            return false;
        }

        id<MTLComputePipelineState> pipeline = [mtlDevice newComputePipelineStateWithFunction:computeFunction error:&error];
        if (!pipeline) {
            printf("MetalEngine: Failed to create compute pipeline: %s\n",
                   [[error localizedDescription] UTF8String]);
            return false;
        }
        computePipeline = (__bridge_retained void*)pipeline;

        // Create initialization kernel
        id<MTLFunction> initFunction = [library newFunctionWithName:@"initializeKangaroos"];
        if (initFunction) {
            id<MTLComputePipelineState> initPipe = [mtlDevice newComputePipelineStateWithFunction:initFunction error:&error];
            if (initPipe) {
                initPipeline = (__bridge_retained void*)initPipe;
            }
        }

        return true;
    }
}

// ---------------------------------------------------------------------------------
// Create Buffers
// ---------------------------------------------------------------------------------

bool MetalEngine::CreateBuffers() {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;

        // Calculate sizes
        kangarooSize = nbThread * GPU_GRP_SIZE * KSIZE * sizeof(uint64_t);
        outputSize = (maxFound * ITEM_SIZE + sizeof(uint32_t));
        jumpSize = NB_JUMP * 8 * 4; // Distance, Px, Py

        // Create Metal buffers with shared storage mode for best CPU-GPU interaction
        MTLResourceOptions options = MTLResourceStorageModeShared;

        // Kangaroo buffer
        id<MTLBuffer> kangBuf = [mtlDevice newBufferWithLength:kangarooSize options:options];
        if (!kangBuf) {
            printf("MetalEngine: Failed to create kangaroo buffer\n");
            return false;
        }
        kangarooBuffer = (__bridge_retained void*)kangBuf;

        // Jump distance buffer (192-bit = 3 uint64_t per jump)
        id<MTLBuffer> jdBuf = [mtlDevice newBufferWithLength:NB_JUMP * 3 * sizeof(uint64_t) options:options];
        if (!jdBuf) {
            printf("MetalEngine: Failed to create jump distance buffer\n");
            return false;
        }
        jumpDistBuffer = (__bridge_retained void*)jdBuf;

        // Jump point X buffer
        id<MTLBuffer> jpxBuf = [mtlDevice newBufferWithLength:NB_JUMP * 4 * sizeof(uint64_t) options:options];
        if (!jpxBuf) {
            printf("MetalEngine: Failed to create jump Px buffer\n");
            return false;
        }
        jumpPxBuffer = (__bridge_retained void*)jpxBuf;

        // Jump point Y buffer
        id<MTLBuffer> jpyBuf = [mtlDevice newBufferWithLength:NB_JUMP * 4 * sizeof(uint64_t) options:options];
        if (!jpyBuf) {
            printf("MetalEngine: Failed to create jump Py buffer\n");
            return false;
        }
        jumpPyBuffer = (__bridge_retained void*)jpyBuf;

        // Found count buffer (atomic counter)
        id<MTLBuffer> fcBuf = [mtlDevice newBufferWithLength:sizeof(uint32_t) options:options];
        if (!fcBuf) {
            printf("MetalEngine: Failed to create found count buffer\n");
            return false;
        }
        foundCountBuffer = (__bridge_retained void*)fcBuf;

        // Output buffer
        id<MTLBuffer> outBuf = [mtlDevice newBufferWithLength:maxFound * sizeof(DPOutput) options:options];
        if (!outBuf) {
            printf("MetalEngine: Failed to create output buffer\n");
            return false;
        }
        outputBuffer = (__bridge_retained void*)outBuf;

        // DP mask buffer
        id<MTLBuffer> dpBuf = [mtlDevice newBufferWithLength:sizeof(uint64_t) options:options];
        if (!dpBuf) {
            printf("MetalEngine: Failed to create DP mask buffer\n");
            return false;
        }
        dpMaskBuffer = (__bridge_retained void*)dpBuf;

        // Max found buffer
        id<MTLBuffer> mfBuf = [mtlDevice newBufferWithLength:sizeof(uint32_t) options:options];
        if (!mfBuf) {
            printf("MetalEngine: Failed to create max found buffer\n");
            return false;
        }
        maxFoundBuffer = (__bridge_retained void*)mfBuf;

        // Host-side memory
        kangarooPinned = (uint64_t*)malloc(nbThreadPerGroup * GPU_GRP_SIZE * KSIZE * sizeof(uint64_t));
        outputPinned = (uint32_t*)malloc(outputSize);
        jumpPinned = (uint64_t*)malloc(jumpSize);

        if (!kangarooPinned || !outputPinned || !jumpPinned) {
            printf("MetalEngine: Failed to allocate host memory\n");
            return false;
        }

        // Initialize max found buffer
        uint32_t* mfPtr = (uint32_t*)[(__bridge id<MTLBuffer>)maxFoundBuffer contents];
        *mfPtr = maxFound;

        return true;
    }
}

// ---------------------------------------------------------------------------------
// Set Parameters (jump points and DP mask)
// ---------------------------------------------------------------------------------

void MetalEngine::SetParams(uint64_t dpMask, Int* distance, Int* px, Int* py) {
    @autoreleasepool {
        this->dpMask = dpMask;

        // Copy DP mask
        uint64_t* dpPtr = (uint64_t*)[(__bridge id<MTLBuffer>)dpMaskBuffer contents];
        *dpPtr = dpMask;

        // Copy jump distances (192-bit = 3 uint64_t per jump)
        uint64_t* jdPtr = (uint64_t*)[(__bridge id<MTLBuffer>)jumpDistBuffer contents];
        for (int i = 0; i < NB_JUMP; i++) {
            jdPtr[i * 3 + 0] = distance[i].bits64[0];
            jdPtr[i * 3 + 1] = distance[i].bits64[1];
            jdPtr[i * 3 + 2] = distance[i].bits64[2];
        }

        // Copy jump point X coordinates
        uint64_t* jpxPtr = (uint64_t*)[(__bridge id<MTLBuffer>)jumpPxBuffer contents];
        for (int i = 0; i < NB_JUMP; i++) {
            jpxPtr[i * 4 + 0] = px[i].bits64[0];
            jpxPtr[i * 4 + 1] = px[i].bits64[1];
            jpxPtr[i * 4 + 2] = px[i].bits64[2];
            jpxPtr[i * 4 + 3] = px[i].bits64[3];
        }

        // Copy jump point Y coordinates
        uint64_t* jpyPtr = (uint64_t*)[(__bridge id<MTLBuffer>)jumpPyBuffer contents];
        for (int i = 0; i < NB_JUMP; i++) {
            jpyPtr[i * 4 + 0] = py[i].bits64[0];
            jpyPtr[i * 4 + 1] = py[i].bits64[1];
            jpyPtr[i * 4 + 2] = py[i].bits64[2];
            jpyPtr[i * 4 + 3] = py[i].bits64[3];
        }
    }
}

// ---------------------------------------------------------------------------------
// Set Kangaroos (initial positions and distances)
// ---------------------------------------------------------------------------------

void MetalEngine::SetKangaroos(Int* px, Int* py, Int* d) {
    if (!initialised || !kangarooBuffer) return;
    @autoreleasepool {
        int gSize = KSIZE * GPU_GRP_SIZE;
        int strideSize = nbThreadPerGroup * KSIZE;
        int nbBlock = nbThread / nbThreadPerGroup;
        int blockSize = nbThreadPerGroup * gSize;
        int idx = 0;

        uint64_t* kangPtr = (uint64_t*)[(__bridge id<MTLBuffer>)kangarooBuffer contents];

        for (int b = 0; b < nbBlock; b++) {
            for (int g = 0; g < GPU_GRP_SIZE; g++) {
                for (int t = 0; t < nbThreadPerGroup; t++) {
                    uint32_t offset = b * blockSize + g * strideSize;

                    // X
                    kangPtr[offset + t + 0 * nbThreadPerGroup] = px[idx].bits64[0];
                    kangPtr[offset + t + 1 * nbThreadPerGroup] = px[idx].bits64[1];
                    kangPtr[offset + t + 2 * nbThreadPerGroup] = px[idx].bits64[2];
                    kangPtr[offset + t + 3 * nbThreadPerGroup] = px[idx].bits64[3];

                    // Y
                    kangPtr[offset + t + 4 * nbThreadPerGroup] = py[idx].bits64[0];
                    kangPtr[offset + t + 5 * nbThreadPerGroup] = py[idx].bits64[1];
                    kangPtr[offset + t + 6 * nbThreadPerGroup] = py[idx].bits64[2];
                    kangPtr[offset + t + 7 * nbThreadPerGroup] = py[idx].bits64[3];

                    // Distance (192-bit = 3 limbs)
                    Int dOff;
                    dOff.Set(&d[idx]);
                    if (idx % 2 == WILD) {
                        dOff.ModAddK1order(&wildOffset);
                    }
                    kangPtr[offset + t + 8 * nbThreadPerGroup] = dOff.bits64[0];
                    kangPtr[offset + t + 9 * nbThreadPerGroup] = dOff.bits64[1];
                    kangPtr[offset + t + 10 * nbThreadPerGroup] = dOff.bits64[2];

#ifdef USE_SYMMETRY
                    // Last jump
                    kangPtr[offset + t + 11 * nbThreadPerGroup] = (uint64_t)NB_JUMP;
#endif

                    idx++;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Get Kangaroos (retrieve current positions and distances)
// ---------------------------------------------------------------------------------

void MetalEngine::GetKangaroos(Int* px, Int* py, Int* d) {
    if (!initialised || !kangarooBuffer) return;
    @autoreleasepool {
        int gSize = KSIZE * GPU_GRP_SIZE;
        int strideSize = nbThreadPerGroup * KSIZE;
        int nbBlock = nbThread / nbThreadPerGroup;
        int blockSize = nbThreadPerGroup * gSize;
        int idx = 0;

        uint64_t* kangPtr = (uint64_t*)[(__bridge id<MTLBuffer>)kangarooBuffer contents];

        for (int b = 0; b < nbBlock; b++) {
            for (int g = 0; g < GPU_GRP_SIZE; g++) {
                for (int t = 0; t < nbThreadPerGroup; t++) {
                    uint32_t offset = b * blockSize + g * strideSize;

                    // X
                    px[idx].bits64[0] = kangPtr[offset + t + 0 * nbThreadPerGroup];
                    px[idx].bits64[1] = kangPtr[offset + t + 1 * nbThreadPerGroup];
                    px[idx].bits64[2] = kangPtr[offset + t + 2 * nbThreadPerGroup];
                    px[idx].bits64[3] = kangPtr[offset + t + 3 * nbThreadPerGroup];
                    px[idx].bits64[4] = 0;

                    // Y
                    py[idx].bits64[0] = kangPtr[offset + t + 4 * nbThreadPerGroup];
                    py[idx].bits64[1] = kangPtr[offset + t + 5 * nbThreadPerGroup];
                    py[idx].bits64[2] = kangPtr[offset + t + 6 * nbThreadPerGroup];
                    py[idx].bits64[3] = kangPtr[offset + t + 7 * nbThreadPerGroup];
                    py[idx].bits64[4] = 0;

                    // Distance (192-bit = 3 limbs)
                    Int dOff;
                    dOff.SetInt32(0);
                    dOff.bits64[0] = kangPtr[offset + t + 8 * nbThreadPerGroup];
                    dOff.bits64[1] = kangPtr[offset + t + 9 * nbThreadPerGroup];
                    dOff.bits64[2] = kangPtr[offset + t + 10 * nbThreadPerGroup];
                    if (idx % 2 == WILD) {
                        dOff.ModSubK1order(&wildOffset);
                    }
                    d[idx].Set(&dOff);

                    idx++;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Set Single Kangaroo
// ---------------------------------------------------------------------------------

void MetalEngine::SetKangaroo(uint64_t kIdx, Int* px, Int* py, Int* d) {
    if (!initialised || !kangarooBuffer) return;
    @autoreleasepool {
        int gSize = KSIZE * GPU_GRP_SIZE;
        int strideSize = nbThreadPerGroup * KSIZE;
        int blockSize = nbThreadPerGroup * gSize;

        uint64_t t = kIdx % nbThreadPerGroup;
        uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
        uint64_t b = kIdx / (nbThreadPerGroup * GPU_GRP_SIZE);

        uint64_t* kangPtr = (uint64_t*)[(__bridge id<MTLBuffer>)kangarooBuffer contents];
        uint32_t offset = b * blockSize + g * strideSize;

        // X
        kangPtr[offset + t + 0 * nbThreadPerGroup] = px->bits64[0];
        kangPtr[offset + t + 1 * nbThreadPerGroup] = px->bits64[1];
        kangPtr[offset + t + 2 * nbThreadPerGroup] = px->bits64[2];
        kangPtr[offset + t + 3 * nbThreadPerGroup] = px->bits64[3];

        // Y
        kangPtr[offset + t + 4 * nbThreadPerGroup] = py->bits64[0];
        kangPtr[offset + t + 5 * nbThreadPerGroup] = py->bits64[1];
        kangPtr[offset + t + 6 * nbThreadPerGroup] = py->bits64[2];
        kangPtr[offset + t + 7 * nbThreadPerGroup] = py->bits64[3];

        // D (192-bit = 3 limbs)
        Int dOff;
        dOff.Set(d);
        if (kIdx % 2 == WILD) {
            dOff.ModAddK1order(&wildOffset);
        }
        kangPtr[offset + t + 8 * nbThreadPerGroup] = dOff.bits64[0];
        kangPtr[offset + t + 9 * nbThreadPerGroup] = dOff.bits64[1];
        kangPtr[offset + t + 10 * nbThreadPerGroup] = dOff.bits64[2];

#ifdef USE_SYMMETRY
        kangPtr[offset + t + 11 * nbThreadPerGroup] = (uint64_t)NB_JUMP;
#endif
    }
}

// ---------------------------------------------------------------------------------
// Set Wild Offset
// ---------------------------------------------------------------------------------

void MetalEngine::SetWildOffset(Int* offset) {
    wildOffset.Set(offset);
}

// ---------------------------------------------------------------------------------
// Call Kernel - Simple synchronous execution with cooperative yielding
// Metal on Apple Silicon supports GPU preemption, but we help by keeping 
// individual kernel invocations short (controlled by NB_RUN in the shader)
// ---------------------------------------------------------------------------------

bool MetalEngine::callKernel() {
    if (!initialised || !computePipeline) return false;

    // Check for shutdown signal
    if (g_metalShutdown.load()) {
        return false;
    }
    
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)computePipeline;

        // Reset found count
        uint32_t* fcPtr = (uint32_t*)[(__bridge id<MTLBuffer>)foundCountBuffer contents];
        *fcPtr = 0;

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            printf("MetalEngine: Failed to create command buffer\n");
            return false;
        }

        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            printf("MetalEngine: Failed to create compute encoder\n");
            return false;
        }

        // Set pipeline state
        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:(__bridge id<MTLBuffer>)kangarooBuffer offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpDistBuffer offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpPxBuffer offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpPyBuffer offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)foundCountBuffer offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outputBuffer offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)dpMaskBuffer offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)maxFoundBuffer offset:0 atIndex:7];

        // Calculate grid size
        NSUInteger threadsPerGroup = nbThreadPerGroup;
        NSUInteger numGroups = nbThread / nbThreadPerGroup;

        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake(numGroups, 1, 1);

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            printf("MetalEngine: Kernel execution failed: %s\n",
                   [[commandBuffer.error localizedDescription] UTF8String]);
            return false;
        }

        return true;
    }
}

// ---------------------------------------------------------------------------------
// Call Kernel and Wait
// ---------------------------------------------------------------------------------

bool MetalEngine::callKernelAndWait() {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)computePipeline;

        // Reset found count
        uint32_t* fcPtr = (uint32_t*)[(__bridge id<MTLBuffer>)foundCountBuffer contents];
        *fcPtr = 0;

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            printf("MetalEngine: Failed to create command buffer\n");
            return false;
        }

        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            printf("MetalEngine: Failed to create compute encoder\n");
            return false;
        }

        // Set pipeline state
        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:(__bridge id<MTLBuffer>)kangarooBuffer offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpDistBuffer offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpPxBuffer offset:0 atIndex:2];
        [encoder setBuffer:(__bridge id<MTLBuffer>)jumpPyBuffer offset:0 atIndex:3];
        [encoder setBuffer:(__bridge id<MTLBuffer>)foundCountBuffer offset:0 atIndex:4];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outputBuffer offset:0 atIndex:5];
        [encoder setBuffer:(__bridge id<MTLBuffer>)dpMaskBuffer offset:0 atIndex:6];
        [encoder setBuffer:(__bridge id<MTLBuffer>)maxFoundBuffer offset:0 atIndex:7];

        // Calculate grid size
        NSUInteger threadsPerGroup = nbThreadPerGroup;
        NSUInteger numGroups = nbThread / nbThreadPerGroup;

        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake(numGroups, 1, 1);

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error) {
            printf("MetalEngine: Kernel execution failed: %s\n",
                   [[commandBuffer.error localizedDescription] UTF8String]);
            return false;
        }

        return true;
    }
}

// ---------------------------------------------------------------------------------
// Launch Kernel and Get Results
// ---------------------------------------------------------------------------------

bool MetalEngine::Launch(std::vector<ITEM>& hashFound, bool spinWait) {
    if (!initialised || !computePipeline) return false;

    @autoreleasepool {
        hashFound.clear();

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)computePipeline;

        // Get found count
        uint32_t* fcPtr = (uint32_t*)[(__bridge id<MTLBuffer>)foundCountBuffer contents];
        uint32_t nbFound = *fcPtr;

        if (nbFound > maxFound) {
            if (!lostWarning) {
                printf("\nWarning, %d items lost\nHint: Search with fewer threads (-g) or increase dp (-d)\n",
                       (nbFound - maxFound));
                lostWarning = true;
            }
            nbFound = maxFound;
        }

        // Get results
        DPOutput* outputs = (DPOutput*)[(__bridge id<MTLBuffer>)outputBuffer contents];

        for (uint32_t i = 0; i < nbFound; i++) {
            ITEM it;

            it.kIdx = outputs[i].kIdx;

            it.x.bits64[0] = ((uint64_t)outputs[i].x[1] << 32) | outputs[i].x[0];
            it.x.bits64[1] = ((uint64_t)outputs[i].x[3] << 32) | outputs[i].x[2];
            it.x.bits64[2] = ((uint64_t)outputs[i].x[5] << 32) | outputs[i].x[4];
            it.x.bits64[3] = ((uint64_t)outputs[i].x[7] << 32) | outputs[i].x[6];
            it.x.bits64[4] = 0;

            // 192-bit distance
            it.d.bits64[0] = ((uint64_t)outputs[i].dist[1] << 32) | outputs[i].dist[0];
            it.d.bits64[1] = ((uint64_t)outputs[i].dist[3] << 32) | outputs[i].dist[2];
            it.d.bits64[2] = ((uint64_t)outputs[i].dist[5] << 32) | outputs[i].dist[4];
            it.d.bits64[3] = 0;
            it.d.bits64[4] = 0;

            if (it.kIdx % 2 == WILD) {
                it.d.ModSubK1order(&wildOffset);
            }

            hashFound.push_back(it);
        }

        // Start next kernel
        return callKernel();
    }
}

// ---------------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------------

int MetalEngine::GetNbThread() {
    return nbThread;
}

int MetalEngine::GetGroupSize() {
    return GPU_GRP_SIZE;
}

int MetalEngine::GetMemory() {
    return kangarooSize + outputSize + jumpSize;
}

// ---------------------------------------------------------------------------------
// Static Methods
// ---------------------------------------------------------------------------------

void MetalEngine::PrintMetalInfo() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

        if (devices.count == 0) {
            printf("MetalEngine: No Metal devices found\n");
            return;
        }

        for (NSUInteger i = 0; i < devices.count; i++) {
            id<MTLDevice> device = devices[i];
            
            printf("GPU #%lu %s\n", i, [device.name UTF8String]);
            printf("  Max threads per threadgroup: %lu\n", device.maxThreadsPerThreadgroup.width);
            printf("  Recommended max working set: %.1f MB\n", 
                   (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0));
            printf("  Has unified memory: %s\n", device.hasUnifiedMemory ? "Yes" : "No");
            printf("  Low power: %s\n", device.isLowPower ? "Yes" : "No");
        }
    }
}

bool MetalEngine::GetGridSize(int gpuId, int* x, int* y) {
    @autoreleasepool {
        if (*x <= 0 || *y <= 0) {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

            if (devices.count == 0) {
                printf("MetalEngine: No Metal devices found\n");
                return false;
            }

            id<MTLDevice> device = nil;
            if (gpuId >= 0 && gpuId < (int)devices.count) {
                device = devices[gpuId];
            } else {
                device = MTLCreateSystemDefaultDevice();
            }

            if (!device) {
                printf("MetalEngine: Failed to get Metal device\n");
                return false;
            }

            // For Apple Silicon M4 Pro - balance performance vs initialization time
            if (*x <= 0) {
                // Number of threadgroups - 128 for good GPU utilization
                *x = 128;
            }
            if (*y <= 0) {
                // Threads per threadgroup - 128 for full SIMD utilization
                *y = 128;
            }
        }

        return true;
    }
}

void* MetalEngine::AllocatePinnedMemory(size_t size) {
    // On macOS with unified memory, regular malloc is already "pinned"
    return malloc(size);
}

void MetalEngine::FreePinnedMemory(void* buff) {
    free(buff);
}

#endif // __APPLE__
