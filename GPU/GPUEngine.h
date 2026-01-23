/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../Constants.h"
#include "../SECPK1/SECP256k1.h"

// KSIZE: x[4] + y[4] + dist[3] + lastJump[1] = 12 with symmetry, 11 without
#ifdef USE_SYMMETRY
#define KSIZE 12
#else
#define KSIZE 11
#endif

// ITEM_SIZE: x[32] + dist[24] + kIdx[8] = 64 bytes for 192-bit distance
#define ITEM_SIZE   64
#define ITEM_SIZE32 (ITEM_SIZE/4)

typedef struct {
  Int x;
  Int d;
  uint64_t kIdx;
} ITEM;

class GPUEngine {

public:

  GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound);
  ~GPUEngine();
  void SetParams(uint64_t dpMask,Int *distance,Int *px,Int *py);
  void SetKangaroos(Int *px,Int *py,Int *d);
  void GetKangaroos(Int *px,Int *py,Int *d);
  void SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d);
  bool Launch(std::vector<ITEM> &hashFound,bool spinWait = false);
  void SetWildOffset(Int *offset);
  int GetNbThread();
  int GetGroupSize();
  int GetMemory();
  bool callKernelAndWait();
  bool callKernel();

  std::string deviceName;

  static void *AllocatePinnedMemory(size_t size);
  static void FreePinnedMemory(void *buff);
  static void PrintCudaInfo();
  static bool GetGridSize(int gpuId,int *x,int *y);

private:

  Int wildOffset;
  int nbThread;
  int nbThreadPerGroup;
  uint64_t *inputKangaroo;
  uint64_t *inputKangarooPinned;
  uint32_t *outputItem;
  uint32_t *outputItemPinned;
  uint64_t *jumpPinned;
  bool initialised;
  bool lostWarning;
  uint32_t maxFound;
  uint32_t outputSize;
  uint32_t kangarooSize;
  uint32_t kangarooSizePinned;
  uint32_t jumpSize;
  uint64_t dpMask;

};

#endif // GPUENGINEH
