#pragma once
//!
//! DOES NOT CONTAIN TEX/SURF REFERENCE API AS IT IS ONLY USED FOR PRE-3.x CC
//!
//TODO: Mathematical Functions
#ifdef __INTELLISENSE__
#include <cuda_runtime.h>
#include <cstdint>
const int warpSize = 32;
//Two 16-bit floats
typedef unsigned int __half2;
//One 16-bit float
typedef uint16_t __half;
//3 ints accessed with x, y, z
struct _3comp{int x,y,z;};
//uint3 Thread index within block of threads
_3comp threadIdx;
//uint3 Block index within grid of blocks
_3comp blockIdx;
//dim3 Dimensions of grid of blocks
_3comp gridDim;
//dim3 Dimensions of block of threads
_3comp blockDim;
//Waits for all threads in block to reach this point, flushes memory accesses to all threads in block before proceeding
//Use in conditional code is dangerous
void __syncthreads();
//Like __syncthreads, returns number of threads for which predicate was not 0
int __syncthreads_count(int predicate);
//Like __syncthreads, returns non-zero if predicate is non-zero for all threads in block
int __syncthreads_and(int predicate);
//Like __syncthreads, returns non-zero if predicate is non-zero for any thread in block
int __syncthreads_or(int predicate);
//Like __syncthreads, but per-warp lane named in mask.
//All non-exited threads named in mask must execute a corresponding __syncwarp() with the same mask, or the result is undefined
void __syncwarp(unsigned mask=0xffffffff);
//Writes observer in-block, all reads performed before read after func
void __threadfence_block();
//Like __threadfence_block for whole block + all writes performed before write after func
void __threadfence();
//Like __threadfence + all writes observed by all threads on device and peers
void __threadfence_system();
//Read-Only Data Cache Load <br>
//Returns data of type T at address, operation is cached in read-only cache
template<typename T>T __ldg(const T* address);
///Returns Per-SM counter incremented each clock cycle. Updated with time slicing not taken into account
///Can also use clock_t clock() like in C
long long int clock64();
//Atomic add transaction
int atomicAdd(int* address,int val);
//Atomic add transaction
unsigned int atomicAdd(unsigned int* address,unsigned int val);
//Atomic add transaction
unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val);
//Atomic add transaction (CC 2.x+)
float atomicAdd(float* address,float val);
//Atomic add transaction (CC 6.x+)
double atomicAdd(double* address,double val);
//Atomic add transaction (CC 6.x+)
//Not guaranteed to be a single 32-bit access
__half2 atomicAdd(__half2 *address,__half2 val);
//Atomic add transaction (CC 7.x+)
__half atomicAdd(__half *address,__half val);
//Atomic sub transaction
int atomicSub(int* address,int val);
//Atomic sub transaction
unsigned int atomicSub(unsigned int* address,unsigned int val);
//Stores val at address, returns old
int atomicExch(int* address,int val);
//Stores val at address, returns old
unsigned int atomicExch(unsigned int* address,unsigned int val);
//Stores val at address, returns old
unsigned long long int atomicExch(unsigned long long int* address,unsigned long long int val);
//Stores val at address, returns old
float atomicExch(float* address,float val);
//Stores minimum of val and old in address, returns old (64-bit is CC 3.5+)
int atomicMin(int* address,int val);
//Stores minimum of val and old in address, returns old (64-bit is CC 3.5+)
unsigned int atomicMin(unsigned int* address,unsigned int val);
//Stores minimum of val and old in address, returns old (64-bit is CC 3.5+)
unsigned long long int atomicMin(unsigned long long int* address,unsigned long long int val);
//Stores maximum of val and old in address, returns old (64-bit is CC 3.5+)
int atomicMax(int* address,int val);
//Stores maximum of val and old in address, returns old (64-bit is CC 3.5+)
unsigned int atomicMax(unsigned int* address,unsigned int val);
//Stores maximum of val and old in address, returns old (64-bit is CC 3.5+)
unsigned long long int atomicMax(unsigned long long int* address,unsigned long long int val);
//Stores ((old >= val) ? 0 : (old+1)) in address, returns old
unsigned int atomicInc(unsigned int* address,unsigned int val);
//Stores (((old == 0) | (old > val)) ? val : (old-1)) in address, returns old
unsigned int atomicDec(unsigned int* address,unsigned int val);
//Compare And Swap: reads old at address, computes (old == compare ? val : old) and stores at address, returns old
int atomicCAS(int* address,int compare,int val);
//Compare And Swap: reads old at address, computes (old == compare ? val : old) and stores at address, returns old
unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val);
//Compare And Swap: reads old at address, computes (old == compare ? val : old) and stores at address, returns old
unsigned long long int atomicCAS(unsigned long long int* address,unsigned long long int compare,unsigned long long int val);
//Compare And Swap: reads old at address, computes (old == compare ? val : old) and stores at address, returns old
unsigned short int atomicCAS(unsigned short int *address,unsigned short int compare,unsigned short int val);
//\*address = (old & val) (64-bit is CC 3.5+)
int atomicAnd(int* address,int val);
//\*address = (old & val) (64-bit is CC 3.5+)
unsigned int atomicAnd(unsigned int* address,unsigned int val);
//\*address = (old & val) (64-bit is CC 3.5+)
unsigned long long int atomicAnd(unsigned long long int* address,unsigned long long int val);
//\*address = (old | val) (64-bit is CC 3.5+)
int atomicOr(int* address,int val);
//\*address = (old | val) (64-bit is CC 3.5+)
unsigned int atomicOr(unsigned int* address,unsigned int val);
//\*address = (old | val) (64-bit is CC 3.5+)
unsigned long long int atomicOr(unsigned long long int* address,unsigned long long int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
int atomicXor(int* address,int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
unsigned int atomicXor(unsigned int* address,unsigned int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
unsigned long long int atomicXor(unsigned long long int* address,unsigned long long int val);
//Returns 1 if ptr is in global memory, otherwise 0
unsigned int __isGlobal(const void* ptr);
//Returns 1 if ptr is in shared memory, otherwise 0
unsigned int __isShared(const void* ptr);
//Returns 1 if ptr is in constant memory, otherwise 0
unsigned int __isConstant(const void* ptr);
//Returns 1 if ptr is in local memory, otherwise 0
unsigned int __isLocal(const void* ptr);
//Increment per-kernel counter (0 - 7)
void __prof_trigger(int counter);
//Assert that expression is 0, otherwise stop kernel execution and trigger a breakpoint
void assert(int expression);
//Fetch from texObj using int coordinate x, non-normalized coords only, no filtering, only border and clamp addressing
//Can promote integer to float
template<class T>T tex1Dfetch(cudaTextureObject_t texObj,int x);
//Fetch from texObj using float coordinate x
template<class T>T tex1D(cudaTextureObject_t texObj,float x);
//Fetch from texObj using float coordinate x at LOD level
template<class T>T tex1DLod(cudaTextureObject_t texObj,float x,float level);
//Fetch from texObj using float coordinate x, LOD is derived from dx and dy gradient
template<class T>T tex1DGrad(cudaTextureObject_t texObj,float x,float dx,float dy);
//Fetch from texObj using float coordinate x and y
template<class T>T tex2D(cudaTextureObject_t texObj,float x,float y);
//Fetch from texObj using float coordinate x and y at LOD level
template<class T>T tex2DLod(cudaTextureObject_t texObj,float x,float y,float level);
//Fetch from texObj using float coordinate x and y, LOD is derived from dx and dy gradient
template<class T>T tex2DGrad(cudaTextureObject_t texObj,float x,float y,float2 dx,float2 dy);
//Fetch from texObj using float coordinate x, y, and z
template<class T>T tex3D(cudaTextureObject_t texObj,float x,float y,float z);
//Fetch from texObj using float coordinate x, y, and z at LOD level
template<class T>T tex3DLod(cudaTextureObject_t texObj,float x,float y,float z,float level);
//Fetch from texObj using float coordinate x, y, and z, LOD is derived from dx and dy gradient
template<class T>T tex3DGrad(cudaTextureObject_t texObj,float x,float y,float z,float4 dx,float4 dy);
//Fetch from texObj using float coordinate x, extracts from layer layer
template<class T>T tex1DLayered(cudaTextureObject_t texObj,float x,int layer);
//Fetch from texObj using float coordinate x at LOD level, extracts from layer layer
template<class T>T tex1DLayeredLod(cudaTextureObject_t texObj,float x,int layer,float level);
//Fetch from texObj using float coordinate x, LOD is derived from dx and dy gradient, extracts from layer layer
template<class T>T tex1DLayeredGrad(cudaTextureObject_t texObj,float x,int layer,float dx,float dy);
//Fetch from texObj using float coordinate x and y, extracts from layer layer
template<class T>T tex2DLayered(cudaTextureObject_t texObj,float x,float y,int layer);
//Fetch from texObj using float coordinate x and y at LOD level, extracts from layer layer
template<class T>T tex2DLayeredLod(cudaTextureObject_t texObj,float x,float y,int layer,float level);
//Fetch from texObj using float coordinate x and y, LOD is derived from dx and dy gradient, extracts from layer layer
template<class T>T tex2DLayeredGrad(cudaTextureObject_t texObj,float x,float y,int layer,float2 dx,float2 dy);
//Fetch from texObj using coordinate x, y, and z; uses cubemap texture rules
template<class T>T texCubemap(cudaTextureObject_t texObj,float x,float y,float z);
//Fetch from texObj using coordinate x, y, and z; at LOD level; uses cubemap texture rules
template<class T>T texCubemapLod(cudaTextureObject_t texObj,float x,float y,float z,float level);
//Fetch from texObj using coordinate x, y, and z; extracts from layer; uses cubemap texture rules
template<class T>T texCubemapLayered(cudaTextureObject_t texObj,float x,float y,float z,int layer);
//Fetch from texObj using coordinate x, y, and z; extracts from layer; at LOD level; uses cubemap texture rules
template<class T>T texCubemapLayeredLod(cudaTextureObject_t texObj,float x,float y,float z,int layer,float level);
//Fetch from texObj using coordinate x and y, and comp parameter
template<class T>T tex2Dgather(cudaTextureObject_t texObj,float x,float y,int comp=0);
//Read data from 1D surface at coord x with out of range handling method boundaryMode
template<class T>T surf1Dread(cudaSurfaceObject_t surfObj,int x,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 1D surface at coord x with out of range handling method boundaryMode
template<class T>void surf1Dwrite(T data,cudaSurfaceObject_t surfObj,int x,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 2D surface at coord x,y with out of range handling method boundaryMode
template<class T>T surf2Dread(cudaSurfaceObject_t surfObj,int x,int y,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 1D surface at coord x,y with out of range handling method boundaryMode
template<class T>void surf2Dread(T* data,cudaSurfaceObject_t surfObj,int x,int y,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 2D surface at coord x,y  with out of range handling method boundaryMode
template<class T>void surf2Dwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 3D surface at coord x,y,z with out of range handling method boundaryMode
template<class T>T surf3Dread(cudaSurfaceObject_t surfObj,int x,int y,int z,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 3D surface at coord x,y,z with out of range handling method boundaryMode
template<class T>void surf3Dread(T* data,cudaSurfaceObject_t surfObj,int x,int y,int z,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 3D surface at coord x,y,z with out of range handling method boundaryMode
template<class T>void surf3Dwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,int z,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 1D layered surface at coord x from layer with out of range handling method boundaryMode
template<class T>T surf1DLayeredread(cudaSurfaceObject_t surfObj,int x,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 1D layered surface at coord x from layer with out of range handling method boundaryMode
template<class T>void surf1DLayeredread(T data,cudaSurfaceObject_t surfObj,int x,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 1D layered surface at coord x from layer with out of range handling method boundaryMode
template<class T>void surf1DLayeredwrite(T data,cudaSurfaceObject_t surfObj,int x,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 2D layered surface at coord x,y from layer with out of range handling method boundaryMode
template<class T>T surf2DLayeredread(cudaSurfaceObject_t surfObj,int x,int y,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 2D layered surface at coord x,y from layer with out of range handling method boundaryMode
template<class T>void surf2DLayeredread(T data,cudaSurfaceObject_t surfObj,int x,int y,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 2D layered surface at coord x,y from layer with out of range handling method boundaryMode
template<class T>void surf2DLayeredwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,int layer,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from cubemap surface at coord x,y and face index face with out of range handling method boundaryMode
template<class T>T surfCubemapread(cudaSurfaceObject_t surfObj,int x,int y,int face,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from cubemap surface at coord x,y and face index face with out of range handling method boundaryMode
template<class T>void surfCubemapread(T data,cudaSurfaceObject_t surfObj,int x,int y,int face,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to cubemap surface at coord x,y and face index face with out of range handling method boundaryMode
template<class T>void surfCubemapwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,int face,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from cubemap surface at coord x,y and layer face index layerFace with out of range handling method boundaryMode
template<class T>T surfCubemapLayeredread(cudaSurfaceObject_t surfObj,int x,int y,int layerFace,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from cubemap surface at coord x,y and layer face index layerFace with out of range handling method boundaryMode
template<class T>void surfCubemapLayeredread(T data,cudaSurfaceObject_t surfObj,int x,int y,int layerFace,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to cubemap surface at coord x,y and layer face index layerFace with out of range handling method boundaryMode
template<class T>void surfCubemapLayeredwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,int layerFace,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Evaluate predicate for all non-exited threads in mask, return non-zero if predicate is non-zero for all of them
int __all_sync(unsigned mask,int predicate);
//Evaluate predicate for all non-exited threads in mask, return non-zero if predicate is non-zero for any of them
int __any_sync(unsigned mask,int predicate);
//Evaluate predicate for all non-exited threads in mask, return integer with
//Nth bit set if predicate is non-zero for Nth thread of warp when Nth thread is active
unsigned __ballot_sync(unsigned mask,int predicate);
//Returns a 32-bit mask of active threads in calling warp. Nth bit is set if Nth lange in warp is active.
//Inactive threads are 0 bits in returned mask. Exited threads are marked as inactive. Does not synchronize following functions
unsigned __activemask();
//Broadcast-and-compare - returns mask of threads that have the same value in mask (CC 7.x+)
template<typename T>unsigned int __match_any_sync(unsigned mask,T value);
//Broadcast-and-compare - returns mask if all threads in mask have the same value, otherwise 0
//Pred is set to true if all threads in mask have the same value, otherwise false (CC 7.x+)
template<typename T>unsigned int __match_all_sync(unsigned mask,T value,int *pred);
//Direct copy from indexed lane
//Returns the value of var held by the thread whose ID is given by srcLane.
//If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.
//If srcLane is outside the range [0:width-1], the value returned corresponds to the value of var held by the srcLane modulo width
template<typename T>T __shfl_sync(unsigned mask,T var,int srcLane,int width=warpSize);
//Copy from a lane with lower ID relative to called
//Calculates a source lane ID by subtracting delta from the caller's lane ID.
//The value of var held by the resulting lane ID is returned: in effect, var is shifted up the warp by delta lanes.
//If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.
//The source lane index will not wrap around the value of width, so effectively the lower delta lanes will be unchanged.
template<typename T>T __shfl_up_sync(unsigned mask,T var,unsigned int delta,int width=warpSize);
//Copy from a lane with higher ID relative to called
//Same as __shfl_up_sync for higher ID
template<typename T>T __shfl_down_sync(unsigned mask,T var,unsigned int delta,int width=warpSize);
//Copy from a lane based on bitwise XOR of own lane ID
//Calculates a source line ID by performing a bitwise XOR of the caller's lane ID with laneMask: the value of var held by the resulting lane ID is returned.
//If width is less than warpSize then each group of width consecutive threads are able to access elements from earlier groups of threads,
//however if they attempt to access elements from later groups of threads their own value of var will be returned.
//This mode implements a butterfly addressing pattern such as is used in tree reduction and broadcast.
template<typename T> T __shfl_xor_sync(unsigned mask,T var,int laneMask,int width=warpSize);
#endif
