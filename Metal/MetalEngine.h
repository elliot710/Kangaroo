/*
 * Metal Engine for Pollard's Kangaroo Algorithm
 * Based on JeanLucPons/Kangaroo CUDA implementation
 *
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 *
 * Optimized for Apple Silicon (M1/M2/M3) GPUs
 */

#ifndef METALENGINEH
#define METALENGINEH

#include <vector>
#include <string>
#include "../Constants.h"
#include "../SECPK1/SECP256k1.h"

#ifdef __APPLE__

#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif

#define ITEM_SIZE   56
#define ITEM_SIZE32 (ITEM_SIZE/4)

// Output item from Metal kernel
struct ITEM {
    Int x;
    Int d;
    uint64_t kIdx;
};

// Distinguished point output structure (matches Metal shader)
struct DPOutput {
    uint32_t x[8];       // Point X (256-bit)
    uint32_t dist[4];    // Distance (128-bit)
    uint64_t kIdx;       // Kangaroo index
};

class MetalEngine {

public:
    // Constructor and destructor
    MetalEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound);
    ~MetalEngine();

    // Configuration
    void SetParams(uint64_t dpMask, Int* distance, Int* px, Int* py);
    void SetKangaroos(Int* px, Int* py, Int* d);
    void GetKangaroos(Int* px, Int* py, Int* d);
    void SetKangaroo(uint64_t kIdx, Int* px, Int* py, Int* d);
    void SetWildOffset(Int* offset);

    // Execution
    bool Launch(std::vector<ITEM>& hashFound, bool spinWait = false);
    bool callKernel();
    bool callKernelAndWait();

    // Getters
    int GetNbThread();
    int GetGroupSize();
    int GetMemory();

    // Static methods
    static void PrintMetalInfo();
    static void PrintCudaInfo() { PrintMetalInfo(); }  // Alias for compatibility
    static bool GetGridSize(int gpuId, int* x, int* y);
    static void* AllocatePinnedMemory(size_t size);
    static void FreePinnedMemory(void* buff);

    std::string deviceName;
    bool initialised;

private:
    // Metal objects (opaque pointers)
    void* device;              // id<MTLDevice>
    void* commandQueue;        // id<MTLCommandQueue>
    void* computePipeline;     // id<MTLComputePipelineState>
    void* initPipeline;        // id<MTLComputePipelineState>

    // Buffers
    void* kangarooBuffer;      // id<MTLBuffer> - Kangaroo data
    void* jumpDistBuffer;      // id<MTLBuffer> - Jump distances
    void* jumpPxBuffer;        // id<MTLBuffer> - Jump point X
    void* jumpPyBuffer;        // id<MTLBuffer> - Jump point Y
    void* foundCountBuffer;    // id<MTLBuffer> - Found count
    void* outputBuffer;        // id<MTLBuffer> - Output DPs
    void* dpMaskBuffer;        // id<MTLBuffer> - DP mask
    void* maxFoundBuffer;      // id<MTLBuffer> - Max found

    // Host-side copies
    uint64_t* kangarooPinned;
    uint32_t* outputPinned;
    uint64_t* jumpPinned;

    // Configuration
    int nbThread;
    int nbThreadPerGroup;
    uint32_t maxFound;
    uint64_t dpMask;
    Int wildOffset;

    // Sizes
    uint32_t kangarooSize;
    uint32_t outputSize;
    uint32_t jumpSize;

    bool lostWarning;
    
    // Initialize Metal
    bool InitMetal(int gpuId);
    bool CreateBuffers();
    bool CreatePipelines();
};

#endif // __APPLE__

#endif // METALENGINEH
