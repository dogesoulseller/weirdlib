#pragma once
//!
//! DOES NOT CONTAIN TEX/SURF REFERENCE API AS IT IS ONLY USED FOR PRE-3.x CC
//!
#ifdef __INTELLISENSE__
#include <cuda_runtime.h>
#include <cstdint>
const int warpSize=32;
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
//Like __syncthreads, but per-warp lane named in mask
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
template<typename T>T __ldg(const T*address);
///Returns Per-SM counter incremented each clock cycle. Updated with time slicing not taken into account
///Can also use clock_t clock() like in C
long long int clock64();
//Atomic add transaction
int atomicAdd(int*address,int val);
//Atomic add transaction
unsigned int atomicAdd(unsigned int*address,unsigned int val);
//Atomic add transaction
unsigned long long int atomicAdd(unsigned long long int*address,unsigned long long int val);
//Atomic add transaction(CC 2.x+)
float atomicAdd(float*address,float val);
//Atomic add transaction(CC 6.x+)
float atomicAdd(float*address,float val);
//Atomic add transaction(CC 6.x+)
//Not guaranteed to be a single 32-bit access
__half2 atomicAdd(__half2*address,__half2 val);
//Atomic add transaction(CC 7.x+)
__half atomicAdd(__half*address,__half val);
//Atomic sub transaction
int atomicSub(int*address,int val);
//Atomic sub transaction
unsigned int atomicSub(unsigned int*address,unsigned int val);
//Stores val at address, returns old
int atomicExch(int*address,int val);
//Stores val at address, returns old
unsigned int atomicExch(unsigned int*address,unsigned int val);
//Stores val at address, returns old
unsigned long long int atomicExch(unsigned long long int*address,unsigned long long int val);
//Stores val at address, returns old
float atomicExch(float*address,float val);
//Stores minimum of val and old in address, returns old(64-bit is CC 3.5+)
int atomicMin(int*address,int val);
//Stores minimum of val and old in address, returns old(64-bit is CC 3.5+)
unsigned int atomicMin(unsigned int*address,unsigned int val);
//Stores minimum of val and old in address, returns old(64-bit is CC 3.5+)
unsigned long long int atomicMin(unsigned long long int*address,unsigned long long int val);
//Stores maximum of val and old in address, returns old(64-bit is CC 3.5+)
int atomicMax(int*address,int val);
//Stores maximum of val and old in address, returns old(64-bit is CC 3.5+)
unsigned int atomicMax(unsigned int*address,unsigned int val);
//Stores maximum of val and old in address, returns old(64-bit is CC 3.5+)
unsigned long long int atomicMax(unsigned long long int*address,unsigned long long int val);
//Stores((old >= val) ? 0 :(old+1)) in address, returns old
unsigned int atomicInc(unsigned int*address,unsigned int val);
//Stores(((old == 0) |(old > val)) ? val :(old-1)) in address, returns old
unsigned int atomicDec(unsigned int*address,unsigned int val);
//Compare And Swap: reads old at address, computes(old == compare ? val : old) and stores at address, returns old
int atomicCAS(int*address,int compare,int val);
//Compare And Swap: reads old at address, computes(old == compare ? val : old) and stores at address, returns old
unsigned int atomicCAS(unsigned int*address,unsigned int compare,unsigned int val);
//Compare And Swap: reads old at address, computes(old == compare ? val : old) and stores at address, returns old
unsigned long long int atomicCAS(unsigned long long int*address,unsigned long long int compare,unsigned long long int val);
//Compare And Swap: reads old at address, computes(old == compare ? val : old) and stores at address, returns old
unsigned short int atomicCAS(unsigned short int*address,unsigned short int compare,unsigned short int val);
//\*address = (old & val) (64-bit is CC 3.5+)
int atomicAnd(int*address,int val);
//\*address = (old & val) (64-bit is CC 3.5+)
unsigned int atomicAnd(unsigned int*address,unsigned int val);
//\*address = (old & val) (64-bit is CC 3.5+)
unsigned long long int atomicAnd(unsigned long long int*address,unsigned long long int val);
//\*address = (old | val) (64-bit is CC 3.5+)
int atomicOr(int*address,int val);
//\*address = (old | val) (64-bit is CC 3.5+)
unsigned int atomicOr(unsigned int*address,unsigned int val);
//\*address = (old | val) (64-bit is CC 3.5+)
unsigned long long int atomicOr(unsigned long long int*address,unsigned long long int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
int atomicXor(int*address,int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
unsigned int atomicXor(unsigned int*address,unsigned int val);
//\*address = (old ^ val) (64-bit is CC 3.5+)
unsigned long long int atomicXor(unsigned long long int*address,unsigned long long int val);
//Returns 1 if ptr is in global memory, otherwise 0
unsigned int __isGlobal(const void*ptr);
//Returns 1 if ptr is in shared memory, otherwise 0
unsigned int __isShared(const void*ptr);
//Returns 1 if ptr is in constant memory, otherwise 0
unsigned int __isConstant(const void*ptr);
//Returns 1 if ptr is in local memory, otherwise 0
unsigned int __isLocal(const void*ptr);
//Increment per-kernel counter(0 - 7)
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
template<class T>void surf2Dread(T*data,cudaSurfaceObject_t surfObj,int x,int y,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Write data to 2D surface at coord x,y with out of range handling method boundaryMode
template<class T>void surf2Dwrite(T data,cudaSurfaceObject_t surfObj,int x,int y,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 3D surface at coord x,y,z with out of range handling method boundaryMode
template<class T>T surf3Dread(cudaSurfaceObject_t surfObj,int x,int y,int z,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
//Read data from 3D surface at coord x,y,z with out of range handling method boundaryMode
template<class T>void surf3Dread(T*data,cudaSurfaceObject_t surfObj,int x,int y,int z,cudaSurfaceBoundaryMode boundaryMode=cudaBoundaryModeTrap);
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
// Arc cosine
float acos(float x);
// Nonnegative arc hyperbolic cosine
float acosh(float x);
// Arc sine
float asin(float x);
// Arc hyperbolic sine
float asinh(float x);
// Arc tangent
float atan(float x);
// Arc tangent of ratio between y and x
float atan2(float y,float x);
// Arc hyperbolic tangent
float atanh(float x);
// Cube root
float cbrt(float x);
// Ceiling
float ceil(float x);
// Value with given magnitude, copying sign of second value
float copysign(float x,float y);
// Cosine
float cos(float x);
// Hyperbolic cosine
float cosh(float x);
// Cosine x pi
float cospi(float x);
// Regular modified cylindrical Bessel function of order 0
float cyl_bessel_i0(float x);
// Regular modified cylindrical Bessel function of order 1
float cyl_bessel_i1(float x);
//Calculate the error function of the input argument
float erf(float x);
//Calculate the complementary error function of the input argument
float erfc(float x);
//Calculate the inverse complementary error function of the input argument
float erfcinv(float y);
//Calculate the scaled complementary error function of the input argument
float erfcx(float x);
//Calculate the inverse error function of the input argument
float erfinv(float y);
//Calculate the base e exponential of the input argument
float exp(float x);
//Calculate the base 10 exponential of the input argument
float exp10(float x);
//Calculate the base 2 exponential of the input argument
float exp2(float x);
//Calculate the base e exponential of the input argument, minus 1
float expm1(float x);
//Calculate the absolute value of the input argument
float fabs(float x);
//Compute the positive difference between x and y
float fdim(float x,float y);
//Calculate the largest integer less than or equal to x
float floor(float x);
//Compute x * y + z as a single operation
float fma(float x,float y,float z);
//Determine the maximum numeric value of the arguments
float fmax(float x,float y);
//Determine the minimum numeric value of the arguments
float fmin(float x,float y);
//Calculate the floating-point remainder of x / y
float fmod(float x,float y);
//Extract mantissa and exponent of a floating-point value
float frexp(float x,int*nptr);
//Calculate the square root of the sum of squares of two arguments
float hypot(float x,float y);
//Compute the unbiased integer exponent of the argument
int ilogb(float x);
//Determine whether argument is finite
int isfinite(float a);
//Determine whether argument is infinite
int isinf(float a);
//Determine whether argument is a NaN
int isnan(float a);
//Calculate the value of the Bessel function of the first kind of order 0 for the input argument
float j0(float x);
//Calculate the value of the Bessel function of the first kind of order 1 for the input argument
float j1(float x);
//Calculate the value of the Bessel function of the first kind of order n for the input argument
float jn(int n,float x);
//Calculate the value of x ⋅ 2 e x p
float ldexp(float x,int exp);
//Calculate the natural logarithm of the absolute value of the gamma function of the input argument
float lgamma(float x);
//Round input to nearest integer value
long long int llrint(float x);
//Round to nearest integer value
long long int llround(float x);
//Calculate the base e logarithm of the input argument
float log(float x);
//Calculate the base 10 logarithm of the input argument
float log10(float x);
//Calculate the value of l o g e(1 + x)
float log1p(float x);
//Calculate the base 2 logarithm of the input argument
float log2(float x);
//Calculate the floating point representation of the exponent of the input argument
float logb(float x);
//Round input to nearest integer value
long int lrint(float x);
//Round to nearest integer value
long int lround(float x);
//Break down the input argument into fractional and integral parts
float modf(float x,float*iptr);
//Returns "Not a Number" value
float nan(const char*tagp);
//Round the input argument to the nearest integer
float nearbyint(float x);
//Return next representable float-precision floating-point value after argument
float nextafter(float x,float y);
//Calculate the square root of the sum of squares of any number of coordinates
float norm(int dim,const float*t);
//Calculate the square root of the sum of squares of three coordinates of the argument
float norm3d(float a,float b,float c);
//Calculate the square root of the sum of squares of four coordinates of the argument
float norm4d(float a,float b,float c,float d);
//Calculate the standard normal cumulative distribution function
float normcdf(float y);
//Calculate the inverse of the standard normal cumulative distribution function
float normcdfinv(float y);
//Calculate the value of first argument to the power of second argument
float pow(float x,float y);
//Calculate reciprocal cube root function
float rcbrt(float x);
//Compute float-precision floating-point remainder
float remainder(float x,float y);
//Compute float-precision floating-point remainder and part of quotient
float remquo(float x,float y,int*quo);
//Calculate one over the square root of the sum of squares of two arguments
float rhypot(float x,float y);
//Round to nearest integer value in floating-point
float rint(float x);
//Calculate the reciprocal of square root of the sum of squares of any number of coordinates
float rnorm(int dim,const float*t);
//Calculate one over the square root of the sum of squares of three coordinates of the argument
float rnorm3d(float a,float b,float c);
//Calculate one over the square root of the sum of squares of four coordinates of the argument
float rnorm4d(float a,float b,float c,float d);
//Round to nearest integer value in floating-point
float round(float x);
//Calculate the reciprocal of the square root of the input argument
float rsqrt(float x);
//Scale floating-point input by integer power of two
float scalbln(float x,long int n);
//Scale floating-point input by integer power of two
float scalbn(float x,int n);
//Return the sign bit of the input
int signbit(float a);
//Calculate the sine of the input argument
float sin(float x);
//Calculate the sine and cosine of the first input argument
void sincos(float x,float*sptr,float*cptr);
//Calculate the sine and cosine of the first input argument * pi
void sincospi(float x,float*sptr,float*cptr);
//Calculate the hyperbolic sine of the input argument
float sinh(float x);
//Calculate the sine of the input argument * pi
float sinpi(float x);
//Calculate the square root of the input argument
float sqrt(float x);
//Calculate the tangent of the input argument
float tan(float x);
//Calculate the hyperbolic tangent of the input argument
float tanh(float x);
//Calculate the gamma function of the input argument
float tgamma(float x);
//Truncate input argument to the integral part
float trunc(float x);
//Calculate the value of the Bessel function of the second kind of order 0 for the input argument
float y0(float x);
//Calculate the value of the Bessel function of the second kind of order 1 for the input argument
float y1(float x);
//Calculate the value of the Bessel function of the second kind of order n for the input argument
float yn(int n, float x);
// Arc cosine
float acos(float x);
// Nonnegative arc hyperbolic cosine
float acosh(float x);
// Arc sine
float asin(float x);
// Arc hyperbolic sine
float asinh(float x);
// Arc tangent
float atan(float x);
// Arc tangent of ratio between y and x
float atan2(float y,float x);
// Arc hyperbolic tangent
float atanh(float x);
// Cube root
float cbrt(float x);
// Ceiling
float ceil(float x);
// Value with given magnitude, copying sign of second value
float copysign(float x,float y);
// Cosine
float cos(float x);
// Hyperbolic cosine
float cosh(float x);
// Cosine x pi
float cospi(float x);
// Regular modified cylindrical Bessel function of order 0
float cyl_bessel_i0(float x);
// Regular modified cylindrical Bessel function of order 1
float cyl_bessel_i1(float x);
//Calculate the error function of the input argument
float erf(float x);
//Calculate the complementary error function of the input argument
float erfc(float x);
//Calculate the inverse complementary error function of the input argument
float erfcinv(float y);
//Calculate the scaled complementary error function of the input argument
float erfcx(float x);
//Calculate the inverse error function of the input argument
float erfinv(float y);
//Calculate the base e exponential of the input argument
float exp(float x);
//Calculate the base 10 exponential of the input argument
float exp10(float x);
//Calculate the base 2 exponential of the input argument
float exp2(float x);
//Calculate the base e exponential of the input argument, minus 1
float expm1(float x);
//Calculate the absolute value of the input argument
float fabs(float x);
//Compute the positive difference between x and y
float fdim(float x,float y);
//Calculate the largest integer less than or equal to x
float floor(float x);
//Compute x * y + z as a single operation
float fma(float x,float y,float z);
//Determine the maximum numeric value of the arguments
float fmax(float x,float y);
//Determine the minimum numeric value of the arguments
float fmin(float x,float y);
//Calculate the floating-point remainder of x / y
float fmod(float x,float y);
//Extract mantissa and exponent of a floating-point value
float frexp(float x,int*nptr);
//Calculate the square root of the sum of squares of two arguments
float hypot(float x,float y);
//Compute the unbiased integer exponent of the argument
int ilogb(float x);
//Determine whether argument is finite
int isfinite(float a);
//Determine whether argument is infinite
int isinf(float a);
//Determine whether argument is a NaN
int isnan(float a);
//Calculate the value of the Bessel function of the first kind of order 0 for the input argument
float j0(float x);
//Calculate the value of the Bessel function of the first kind of order 1 for the input argument
float j1(float x);
//Calculate the value of the Bessel function of the first kind of order n for the input argument
float jn(int n,float x);
//Calculate the value of x ⋅ 2 e x p
float ldexp(float x,int exp);
//Calculate the natural logarithm of the absolute value of the gamma function of the input argument
float lgamma(float x);
//Round input to nearest integer value
long long int llrint(float x);
//Round to nearest integer value
long long int llround(float x);
//Calculate the base e logarithm of the input argument
float log(float x);
//Calculate the base 10 logarithm of the input argument
float log10(float x);
//Calculate the value of l o g e(1 + x)
float log1p(float x);
//Calculate the base 2 logarithm of the input argument
float log2(float x);
//Calculate the floating point representation of the exponent of the input argument
float logb(float x);
//Round input to nearest integer value
long int lrint(float x);
//Round to nearest integer value
long int lround(float x);
//Break down the input argument into fractional and integral parts
float modf(float x,float*iptr);
//Returns "Not a Number" value
float nan(const char*tagp);
//Round the input argument to the nearest integer
float nearbyint(float x);
//Return next representable float-precision floating-point value after argument
float nextafter(float x,float y);
//Calculate the square root of the sum of squares of any number of coordinates
float norm(int dim,const float*t);
//Calculate the square root of the sum of squares of three coordinates of the argument
float norm3d(float a,float b,float c);
//Calculate the square root of the sum of squares of four coordinates of the argument
float norm4d(float a,float b,float c,float d);
//Calculate the standard normal cumulative distribution function
float normcdf(float y);
//Calculate the inverse of the standard normal cumulative distribution function
float normcdfinv(float y);
//Calculate the value of first argument to the power of second argument
float pow(float x,float y);
//Calculate reciprocal cube root function
float rcbrt(float x);
//Compute float-precision floating-point remainder
float remainder(float x,float y);
//Compute float-precision floating-point remainder and part of quotient
float remquo(float x,float y,int*quo);
//Calculate one over the square root of the sum of squares of two arguments
float rhypot(float x,float y);
//Round to nearest integer value in floating-point
float rint(float x);
//Calculate the reciprocal of square root of the sum of squares of any number of coordinates
float rnorm(int dim,const float*t);
//Calculate one over the square root of the sum of squares of three coordinates of the argument
float rnorm3d(float a,float b,float c);
//Calculate one over the square root of the sum of squares of four coordinates of the argument
float rnorm4d(float a,float b,float c,float d);
//Round to nearest integer value in floating-point
float round(float x);
//Calculate the reciprocal of the square root of the input argument
float rsqrt(float x);
//Scale floating-point input by integer power of two
float scalbln(float x, long int n);
//Scale floating-point input by integer power of two
float scalbn(float x,int n);
//Return the sign bit of the input
int signbit(float a);
//Calculate the sine of the input argument
float sin(float x);
//Calculate the sine and cosine of the first input argument
void sincos(float x,float*sptr,float*cptr);
//Calculate the sine and cosine of the first input argument * pi
void sincospi(float x,float*sptr,float*cptr);
//Calculate the hyperbolic sine of the input argument
float sinh(float x);
//Calculate the sine of the input argument * pi
float sinpi(float x);
//Calculate the square root of the input argument
float sqrt(float x);
//Calculate the tangent of the input argument
float tan(float x);
//Calculate the hyperbolic tangent of the input argument
float tanh(float x);
//Calculate the gamma function of the input argument
float tgamma(float x);
//Truncate input argument to the integral part
float trunc(float x);
//Calculate the value of the Bessel function of the second kind of order 0 for the input argument
float y0(float x);
//Calculate the value of the Bessel function of the second kind of order 1 for the input argument
float y1(float x);
//Calculate the value of the Bessel function of the second kind of order n for the input argument
float yn(int n, float x);
//Calculate the arc cosine of the input argument
float acosf(float x);
//Calculate the nonnegative arc hyperbolic cosine of the input argument
float acoshf(float x);
//Calculate the arc sine of the input argument
float asinf(float x);
//Calculate the arc hyperbolic sine of the input argument
float asinhf(float x);
//Calculate the arc tangent of the ratio of first and second input arguments
float atan2f(float y,float x);
//Calculate the arc tangent of the input argument
float atanf(float x);
//Calculate the arc hyperbolic tangent of the input argument
float atanhf(float x);
//Calculate the cube root of the input argument
float cbrtf(float x);
//Calculate ceiling of the input argument
float ceilf(float x);
//Create value with given magnitude, copying sign of second value
float copysignf(float x,float y);
//Calculate the cosine of the input argument
float cosf(float x);
//Calculate the hyperbolic cosine of the input argument
float coshf(float x);
//Calculate the cosine of the input argument * pi
float cospif(float x);
//Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument
float cyl_bessel_i0f(float x);
//Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument
float cyl_bessel_i1f(float x);
//Calculate the complementary error function of the input argument
float erfcf(float x);
//Calculate the inverse complementary error function of the input argument
float erfcinvf(float y);
//Calculate the scaled complementary error function of the input argument
float erfcxf(float x);
//Calculate the error function of the input argument
float erff(float x);
//Calculate the inverse error function of the input argument
float erfinvf(float y);
//Calculate the base 10 exponential of the input argument
float exp10f(float x);
//Calculate the base 2 exponential of the input argument
float exp2f(float x);
//Calculate the base e exponential of the input argument
float expf(float x);
//Calculate the base e exponential of the input argument, minus 1
float expm1f(float x);
//Calculate the absolute value of its argument
float fabsf(float x);
//Compute the positive difference between x and y
float fdimf(float x,float y);
//Divide two floating point values
float fdividef(float x,float y);
//Calculate the largest integer less than or equal to x
float floorf(float x);
//Compute x * y + z as a single operation
float fmaf(float x,float y,float z);
//Determine the maximum numeric value of the arguments
float fmaxf(float x,float y);
//Determine the minimum numeric value of the arguments
float fminf(float x,float y);
//Calculate the floating-point remainder of x / y
float fmodf(float x,float y);
//Extract mantissa and exponent of a floating-point value
float frexpf(float x,int*nptr);
//Calculate the square root of the sum of squares of two arguments
float hypotf(float x,float y);
//Compute the unbiased integer exponent of the argument
int ilogbf(float x);
//Determine whether argument is finite
int isfinite(float a);
//Determine whether argument is infinite
int isinf(float a);
//Determine whether argument is a NaN
int isnan(float a);
//Calculate the value of the Bessel function of the first kind of order 0 for the input argument
float j0f(float x);
//Calculate the value of the Bessel function of the first kind of order 1 for the input argument
float j1f(float x);
//Calculate the value of the Bessel function of the first kind of order n for the input argument
float jnf(int n,float x);
//Calculate the value of x ⋅ 2 e x p
float ldexpf(float x,int exp);
//Calculate the natural logarithm of the absolute value of the gamma function of the input argument
float lgammaf(float x);
//Round input to nearest integer value
long long int llrintf(float x);
//Round to nearest integer value
long long int llroundf(float x);
//Calculate the base 10 logarithm of the input argument
float log10f(float x);
//Calculate the value of l o g e(1 + x)
float log1pf(float x);
//Calculate the base 2 logarithm of the input argument
float log2f(float x);
//Calculate the floating point representation of the exponent of the input argument
float logbf(float x);
//Calculate the natural logarithm of the input argument
float logf(float x);
//Round input to nearest integer value
long int lrintf(float x);
//Round to nearest integer value
long int lroundf(float x);
//Break down the input argument into fractional and integral parts
float modff(float x,float*iptr);
//Returns "Not a Number" value
float nanf(const char*tagp);
//Round the input argument to the nearest integer
float nearbyintf(float x);
//Return next representable single-precision floating-point value afer argument
float nextafterf(float x,float y);
//Calculate the square root of the sum of squares of three coordinates of the argument
float norm3df(float a,float b,float c);
//Calculate the square root of the sum of squares of four coordinates of the argument
float norm4df(float a,float b,float c,float d);
//Calculate the standard normal cumulative distribution function
float normcdff(float y);
//Calculate the inverse of the standard normal cumulative distribution function
float normcdfinvf(float y);
//Calculate the square root of the sum of squares of any number of coordinates
float normf(int dim,const float*a);
//Calculate the value of first argument to the power of second argument
float powf(float x,float y);
//Calculate reciprocal cube root function
float rcbrtf(float x);
//Compute single-precision floating-point remainder
float remainderf(float x,float y);
//Compute single-precision floating-point remainder and part of quotient
float remquof(float x,float y,int*quo);
//Calculate one over the square root of the sum of squares of two arguments
float rhypotf(float x,float y);
//Round input to nearest integer value in floating-point
float rintf(float x);
//Calculate one over the square root of the sum of squares of three coordinates of the argument
float rnorm3df(float a,float b,float c);
//Calculate one over the square root of the sum of squares of four coordinates of the argument
float rnorm4df(float a,float b,float c,float d);
//Calculate the reciprocal of square root of the sum of squares of any number of coordinates
float rnormf(int dim,const float*a);
//Round to nearest integer value in floating-point
float roundf(float x);
//Calculate the reciprocal of the square root of the input argument
float rsqrtf(float x);
//Scale floating-point input by integer power of two
float scalblnf(float x,long int n);
//Scale floating-point input by integer power of two
float scalbnf(float x,int n);
//Return the sign bit of the input
int signbit(float a);
//Calculate the sine and cosine of the first input argument
void sincosf(float x,float*sptr,float*cptr);
//Calculate the sine and cosine of the first input argument * pi
void sincospif(float x,float*sptr,float*cptr);
//Calculate the sine of the input argument
float sinf(float x);
//Calculate the hyperbolic sine of the input argument
float sinhf(float x);
//Calculate the sine of the input argument * pi
float sinpif(float x);
//Calculate the square root of the input argument
float sqrtf(float x);
//Calculate the tangent of the input argument
float tanf(float x);
//Calculate the hyperbolic tangent of the input argument
float tanhf(float x);
//Calculate the gamma function of the input argument
float tgammaf(float x);
//Truncate input argument to the integral part
float truncf(float x);
//Calculate the value of the Bessel function of the second kind of order 0 for the input argument
float y0f(float x);
//Calculate the value of the Bessel function of the second kind of order 1 for the input argument
float y1f(float x);
//Calculate the value of the Bessel function of the second kind of order n for the input argument
float ynf(int n,float x);
//Calculate the fast approximate cosine of the input argument
float __cosf(float x);
//Calculate the fast approximate base 10 exponential of the input argument
float __exp10f(float x);
//Calculate the fast approximate base e exponential of the input argument
float __expf(float x);
//Add two floating point values in round-down mode
float __fadd_rd(float x,float y);
//Add two floating point values in round-to-nearest-even mode
float __fadd_rn(float x,float y);
//Add two floating point values in round-up mode
float __fadd_ru(float x,float y);
//Add two floating point values in round-towards-zero mode
float __fadd_rz(float x,float y);
//Divide two floating point values in round-down mode
float __fdiv_rd(float x,float y);
//Divide two floating point values in round-to-nearest-even mode
float __fdiv_rn(float x,float y);
//Divide two floating point values in round-up mode
float __fdiv_ru(float x,float y);
//Divide two floating point values in round-towards-zero mode
float __fdiv_rz(float x,float y);
//Calculate the fast approximate division of the input arguments
float __fdividef(float x,float y);
//Compute x * y + z as a single operation,in round-down mode
float __fmaf_rd(float x,float y,float z);
//Compute x * y + z as a single operation,in round-to-nearest-even mode
float __fmaf_rn(float x,float y,float z);
//Compute x * y + z as a single operation,in round-up mode
float __fmaf_ru(float x,float y,float z);
//Compute x * y + z as a single operation,in round-towards-zero mode
float __fmaf_rz(float x,float y,float z);
//Multiply two floating point values in round-down mode
float __fmul_rd(float x,float y);
//Multiply two floating point values in round-to-nearest-even mode
float __fmul_rn(float x,float y);
//Multiply two floating point values in round-up mode
float __fmul_ru(float x,float y);
//Multiply two floating point values in round-towards-zero mode
float __fmul_rz(float x,float y);
//Compute 1 x in round-down mode
float __frcp_rd(float x);
//Compute 1 x in round-to-nearest-even mode
float __frcp_rn(float x);
//Compute 1 x in round-up mode
float __frcp_ru(float x);
//Compute 1 x in round-towards-zero mode
float __frcp_rz(float x);
//Compute 1 / x in round-to-nearest-even mode
float __frsqrt_rn(float x);
//Compute x in round-down mode
float __fsqrt_rd(float x);
//Compute x in round-to-nearest-even mode
float __fsqrt_rn(float x);
//Compute x in round-up mode
float __fsqrt_ru(float x);
//Compute x in round-towards-zero mode
float __fsqrt_rz(float x);
//Subtract two floating point values in round-down mode
float __fsub_rd(float x,float y);
//Subtract two floating point values in round-to-nearest-even mode
float __fsub_rn(float x,float y);
//Subtract two floating point values in round-up mode
float __fsub_ru(float x,float y);
//Subtract two floating point values in round-towards-zero mode
float __fsub_rz(float x,float y);
//Calculate the fast approximate base 10 logarithm of the input argument
float __log10f(float x);
//Calculate the fast approximate base 2 logarithm of the input argument
float __log2f(float x);
//Calculate the fast approximate base e logarithm of the input argument
float __logf(float x);
//Calculate the fast approximate of x y
float __powf(float x,float y);
//Clamp the input argument to [+0.0,1.0]
float __saturatef(float x);
//Calculate the fast approximate of sine and cosine of the first input argument
void __sincosf(float x,float*sptr,float*cptr);
//Calculate the fast approximate sine of the input argument
float __sinf(float x);
//Calculate the fast approximate tangent of the input argument
float __tanf(float x);
//Add two floating point values in round-down mode
float __dadd_rd(float x,float y);
//Add two floating point values in round-to-nearest-even mode
float __dadd_rn(float x,float y);
//Add two floating point values in round-up mode
float __dadd_ru(float x,float y);
//Add two floating point values in round-towards-zero mode
float __dadd_rz(float x,float y);
//Divide two floating point values in round-down mode
float __ddiv_rd(float x,float y);
//Divide two floating point values in round-to-nearest-even mode
float __ddiv_rn(float x,float y);
//Divide two floating point values in round-up mode
float __ddiv_ru(float x,float y);
//Divide two floating point values in round-towards-zero mode
float __ddiv_rz(float x,float y);
//Multiply two floating point values in round-down mode
float __dmul_rd(float x,float y);
//Multiply two floating point values in round-to-nearest-even mode
float __dmul_rn(float x,float y);
//Multiply two floating point values in round-up mode
float __dmul_ru(float x,float y);
//Multiply two floating point values in round-towards-zero mode
float __dmul_rz(float x,float y);
//Compute 1 x in round-down mode
float __drcp_rd(float x);
//Compute 1 x in round-to-nearest-even mode
float __drcp_rn(float x);
//Compute 1 x in round-up mode
float __drcp_ru(float x);
//Compute 1 x in round-towards-zero mode
float __drcp_rz(float x);
//Compute x in round-down mode
float __dsqrt_rd(float x);
//Compute x in round-to-nearest-even mode
float __dsqrt_rn(float x);
//Compute x in round-up mode
float __dsqrt_ru(float x);
//Compute x in round-towards-zero mode
float __dsqrt_rz(float x);
//Subtract two floating point values in round-down mode
float __dsub_rd(float x,float y);
//Subtract two floating point values in round-to-nearest-even mode
float __dsub_rn(float x,float y);
//Subtract two floating point values in round-up mode
float __dsub_ru(float x,float y);
//Subtract two floating point values in round-towards-zero mode
float __dsub_rz(float x,float y);
//Compute x * y + z as a single operation in round-down mode
float __fma_rd(float x,float y,float z);
//Compute x * y + z as a single operation in round-to-nearest-even mode
float __fma_rn(float x,float y,float z);
//Compute x * y + z as a single operation in round-up mode
float __fma_ru(float x,float y,float z);
//Compute x * y + z as a single operation in round-towards-zero mode
float __fma_rz(float x,float y,float z);
//Reverse the bit order of a 32 bit unsigned integer
unsigned int __brev(unsigned int x);
//Reverse the bit order of a 64 bit unsigned integer
unsigned long long int __brevll(unsigned long long int x);
//Return selected bytes from two 32 bit unsigned integers
unsigned int __byte_perm(unsigned int x,unsigned int y,unsigned int s);
//Return the number of consecutive high-order zero bits in a 32 bit integer
int __clz(int x);
//Count the number of consecutive high-order zero bits in a 64 bit integer
int __clzll(long long int x);
//Find the position of the least significant bit set to 1 in a 32 bit integer
int __ffs(int x);
//Find the position of the least significant bit set to 1 in a 64 bit integer
int __ffsll(long long int x);
//Concatenate hi : lo, shift left by shift & 31 bits, return the most significant 32 bits
unsigned int __funnelshift_l(unsigned int lo,unsigned int hi,unsigned int shift);
//Concatenate hi : lo, shift left by min(shift, 32) bits, return the most significant 32 bits
unsigned int __funnelshift_lc(unsigned int lo,unsigned int hi,unsigned int shift);
//Concatenate hi : lo, shift right by shift & 31 bits, return the least significant 32 bits
unsigned int __funnelshift_r(unsigned int lo,unsigned int hi,unsigned int shift);
//Concatenate hi : lo, shift right by min(shift, 32) bits, return the least significant 32 bits
unsigned int __funnelshift_rc(unsigned int lo,unsigned int hi,unsigned int shift);
//Compute average of signed input arguments, avoiding overflow in the intermediate sum
int __hadd(int ,int);
//Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers
int __mul24(int x,int y);
//Calculate the most significant 64 bits of the product of the two 64 bit integers
long long int __mul64hi(long long int x, long long int y);
//Calculate the most significant 32 bits of the product of the two 32 bit integers
int __mulhi(int x,int y);
//Count the number of bits that are set to 1 in a 32 bit integer
int __popc(unsigned int x);
//Count the number of bits that are set to 1 in a 64 bit integer
int __popcll(unsigned long long int x);
//Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum
int __rhadd(int ,int);
//Calculate | x − y | + z , the sum of absolute difference
unsigned int __sad(int x,int y,unsigned int z);
//Compute average of unsigned input arguments, avoiding overflow in the intermediate sum
unsigned int __uhadd(unsigned int,unsigned int);
//Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers
unsigned int __umul24(unsigned int x,unsigned int y);
//Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers
unsigned long long int __umul64hi(unsigned long long int x,unsigned long long int y);
//Calculate the most significant 32 bits of the product of the two 32 bit unsigned integers
unsigned int __umulhi(unsigned int x,unsigned int y);
//Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum
unsigned int __urhadd(unsigned int,unsigned int);
//Calculate | x − y | + z , the sum of absolute difference
unsigned int __usad(unsigned int x,unsigned int y,unsigned int z);
//Convert a float to a float in round-down mode
float __double2float_rd(float x);
//Convert a float to a float in round-to-nearest-even mode
float __double2float_rn(float x);
//Convert a float to a float in round-up mode
float __double2float_ru(float x);
//Convert a float to a float in round-towards-zero mode
float __double2float_rz(float x);
//Reinterpret high 32 bits in a float as a signed integer
int __double2hiint(float x);
//Convert a float to a signed int in round-down mode
int __double2int_rd(float x);
//Convert a float to a signed int in round-to-nearest-even mode
int __double2int_rn(float x);
//Convert a float to a signed int in round-up mode
int __double2int_ru(float x);
//Convert a float to a signed int in round-towards-zero mode
int __double2int_rz(float);
//Convert a float to a signed 64-bit int in round-down mode
long long int __double2ll_rd(float x);
//Convert a float to a signed 64-bit int in round-to-nearest-even mode
long long int __double2ll_rn(float x);
//Convert a float to a signed 64-bit int in round-up mode
long long int __double2ll_ru(float x);
//Convert a float to a signed 64-bit int in round-towards-zero mode
long long int __double2ll_rz(float);
//Reinterpret low 32 bits in a float as a signed integer
int __double2loint(float x);
//Convert a float to an unsigned int in round-down mode
unsigned int __double2uint_rd(float x);
//Convert a float to an unsigned int in round-to-nearest-even mode
unsigned int __double2uint_rn(float x);
//Convert a float to an unsigned int in round-up mode
unsigned int __double2uint_ru(float x);
//Convert a float to an unsigned int in round-towards-zero mode
unsigned int __double2uint_rz(float);
//Convert a float to an unsigned 64-bit int in round-down mode
unsigned long long int __double2ull_rd(float x);
//Convert a float to an unsigned 64-bit int in round-to-nearest-even mode
unsigned long long int __double2ull_rn(float x);
//Convert a float to an unsigned 64-bit int in round-up mode
unsigned long long int __double2ull_ru(float x);
//Convert a float to an unsigned 64-bit int in round-towards-zero mode
unsigned long long int __double2ull_rz(float);
//Reinterpret bits in a float as a 64-bit signed integer
long long int __double_as_longlong(float x);
//Convert a float to a signed integer in round-down mode
int __float2int_rd(float x);
//Convert a float to a signed integer in round-to-nearest-even mode
int __float2int_rn(float x);
//Convert a float to a signed integer in round-up mode
int __float2int_ru(float x);
//Convert a float to a signed integer in round-towards-zero mode
int __float2int_rz(float x);
//Convert a float to a signed 64-bit integer in round-down mode
long long int __float2ll_rd(float x);
//Convert a float to a signed 64-bit integer in round-to-nearest-even mode
long long int __float2ll_rn(float x);
//Convert a float to a signed 64-bit integer in round-up mode
long long int __float2ll_ru(float x);
//Convert a float to a signed 64-bit integer in round-towards-zero mode
long long int __float2ll_rz(float x);
//Convert a float to an unsigned integer in round-down mode
unsigned int __float2uint_rd(float x);
//Convert a float to an unsigned integer in round-to-nearest-even mode
unsigned int __float2uint_rn(float x);
//Convert a float to an unsigned integer in round-up mode
unsigned int __float2uint_ru(float x);
//Convert a float to an unsigned integer in round-towards-zero mode
unsigned int __float2uint_rz(float x);
//Convert a float to an unsigned 64-bit integer in round-down mode
unsigned long long int __float2ull_rd(float x);
//Convert a float to an unsigned 64-bit integer in round-to-nearest-even mode
unsigned long long int __float2ull_rn(float x);
//Convert a float to an unsigned 64-bit integer in round-up mode
unsigned long long int __float2ull_ru(float x);
//Convert a float to an unsigned 64-bit integer in round-towards-zero mode
unsigned long long int __float2ull_rz(float x);
//Reinterpret bits in a float as a signed integer
int __float_as_int(float x);
//Reinterpret bits in a float as a unsigned integer
unsigned int __float_as_uint(float x);
//Reinterpret high and low 32-bit integer values as a float
float __hiloint2double(int hi,int lo);
//Convert a signed int to a float
float __int2double_rn(int x);
//Convert a signed integer to a float in round-down mode
float __int2float_rd(int x);
//Convert a signed integer to a float in round-to-nearest-even mode
float __int2float_rn(int x);
//Convert a signed integer to a float in round-up mode
float __int2float_ru(int x);
//Convert a signed integer to a float in round-towards-zero mode
float __int2float_rz(int x);
//Reinterpret bits in an integer as a float
float __int_as_float(int x);
//Convert a signed 64-bit int to a float in round-down mode
float __ll2double_rd(long long int x);
//Convert a signed 64-bit int to a float in round-to-nearest-even mode
float __ll2double_rn(long long int x);
//Convert a signed 64-bit int to a float in round-up mode
float __ll2double_ru(long long int x);
//Convert a signed 64-bit int to a float in round-towards-zero mode
float __ll2double_rz(long long int x);
//Convert a signed integer to a float in round-down mode
float __ll2float_rd(long long int x);
//Convert a signed 64-bit integer to a float in round-to-nearest-even mode
float __ll2float_rn(long long int x);
//Convert a signed integer to a float in round-up mode
float __ll2float_ru(long long int x);
//Convert a signed integer to a float in round-towards-zero mode
float __ll2float_rz(long long int x);
//Reinterpret bits in a 64-bit signed integer as a float
float __longlong_as_double(long long int x);
//Convert an unsigned int to a float
float __uint2double_rn(unsigned int x);
//Convert an unsigned integer to a float in round-down mode
float __uint2float_rd(unsigned int x);
//Convert an unsigned integer to a float in round-to-nearest-even mode
float __uint2float_rn(unsigned int x);
//Convert an unsigned integer to a float in round-up mode
float __uint2float_ru(unsigned int x);
//Convert an unsigned integer to a float in round-towards-zero mode
float __uint2float_rz(unsigned int x);
//Reinterpret bits in an unsigned integer as a float
float __uint_as_float(unsigned int x);
//Convert an unsigned 64-bit int to a float in round-down mode
float __ull2double_rd(unsigned long long int x);
//Convert an unsigned 64-bit int to a float in round-to-nearest-even mode
float __ull2double_rn(unsigned long long int x);
//Convert an unsigned 64-bit int to a float in round-up mode
float __ull2double_ru(unsigned long long int x);
//Convert an unsigned 64-bit int to a float in round-towards-zero mode
float __ull2double_rz(unsigned long long int x);
//Convert an unsigned integer to a float in round-down mode
float __ull2float_rd(unsigned long long int x);
//Convert an unsigned integer to a float in round-to-nearest-even mode
float __ull2float_rn(unsigned long long int x);
//Convert an unsigned integer to a float in round-up mode
float __ull2float_ru(unsigned long long int x);
//Convert an unsigned integer to a float in round-towards-zero mode
float __ull2float_rz(unsigned long long int x);
//Computes per-halfword absolute value
unsigned int __vabs2(unsigned int a);
//Computes per-byte absolute value
unsigned int __vabs4(unsigned int a);
//Computes per-halfword sum of absolute difference of signed integer
unsigned int __vabsdiffs2(unsigned int a,unsigned int b);
//Computes per-byte absolute difference of signed integer
unsigned int __vabsdiffs4(unsigned int a,unsigned int b);
//Performs per-halfword absolute difference of unsigned integer computation: |a - b|
unsigned int __vabsdiffu2(unsigned int a,unsigned int b);
//Computes per-byte absolute difference of unsigned integer
unsigned int __vabsdiffu4(unsigned int a,unsigned int b);
//Computes per-halfword absolute value with signed saturation
unsigned int __vabsss2(unsigned int a);
//Computes per-byte absolute value with signed saturation
unsigned int __vabsss4(unsigned int a);
//Performs per-halfword (un)signed addition, with wrap-around: a + b
unsigned int __vadd2(unsigned int a,unsigned int b);
//Performs per-byte (un)signed addition
unsigned int __vadd4(unsigned int a,unsigned int b);
//Performs per-halfword addition with signed saturation
unsigned int __vaddss2(unsigned int a,unsigned int b);
//Performs per-byte addition with signed saturation
unsigned int __vaddss4(unsigned int a,unsigned int b);
//Performs per-halfword addition with unsigned saturation
unsigned int __vaddus2(unsigned int a,unsigned int b);
//Performs per-byte addition with unsigned saturation
unsigned int __vaddus4(unsigned int a,unsigned int b);
//Performs per-halfword signed rounded average computation
unsigned int __vavgs2(unsigned int a,unsigned int b);
//Computes per-byte signed rounder average
unsigned int __vavgs4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned rounded average computation
unsigned int __vavgu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned rounded average
unsigned int __vavgu4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed comparison
unsigned int __vcmpeq2(unsigned int a,unsigned int b);
//Performs per-byte (un)signed comparison
unsigned int __vcmpeq4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison: a >= b ? 0xffff : 0
unsigned int __vcmpges2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vcmpges4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0
unsigned int __vcmpgeu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vcmpgeu4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison: a > b ? 0xffff : 0
unsigned int __vcmpgts2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vcmpgts4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison: a > b ? 0xffff : 0
unsigned int __vcmpgtu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vcmpgtu4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison: a <= b ? 0xffff : 0
unsigned int __vcmples2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vcmples4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0
unsigned int __vcmpleu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vcmpleu4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison: a < b ? 0xffff : 0
unsigned int __vcmplts2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vcmplts4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison: a < b ? 0xffff : 0
unsigned int __vcmpltu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vcmpltu4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0
unsigned int __vcmpne2(unsigned int a,unsigned int b);
//Performs per-byte (un)signed comparison
unsigned int __vcmpne4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned average computation
unsigned int __vhaddu2(unsigned int a,unsigned int b);
//Computes per-byte unsigned average
unsigned int __vhaddu4(unsigned int a,unsigned int b);
//Performs per-halfword signed maximum computation
unsigned int __vmaxs2(unsigned int a,unsigned int b);
//Computes per-byte signed maximum
unsigned int __vmaxs4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned maximum computation
unsigned int __vmaxu2(unsigned int a,unsigned int b);
//Computes per-byte unsigned maximum
unsigned int __vmaxu4(unsigned int a,unsigned int b);
//Performs per-halfword signed minimum computation
unsigned int __vmins2(unsigned int a,unsigned int b);
//Computes per-byte signed minimum
unsigned int __vmins4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned minimum computation
unsigned int __vminu2(unsigned int a,unsigned int b);
//Computes per-byte unsigned minimum
unsigned int __vminu4(unsigned int a,unsigned int b);
//Computes per-halfword negation
unsigned int __vneg2(unsigned int a);
//Performs per-byte negation
unsigned int __vneg4(unsigned int a);
//Computes per-halfword negation with signed saturation
unsigned int __vnegss2(unsigned int a);
//Performs per-byte negation with signed saturation
unsigned int __vnegss4(unsigned int a);
//Performs per-halfword sum of absolute difference of signed
unsigned int __vsads2(unsigned int a,unsigned int b);
//Computes per-byte sum of abs difference of signed
unsigned int __vsads4(unsigned int a,unsigned int b);
//Computes per-halfword sum of abs diff of unsigned
unsigned int __vsadu2(unsigned int a,unsigned int b);
//Computes per-byte sum af abs difference of unsigned
unsigned int __vsadu4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed comparison
unsigned int __vseteq2(unsigned int a,unsigned int b);
//Performs per-byte (un)signed comparison
unsigned int __vseteq4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison
unsigned int __vsetges2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vsetges4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned minimum unsigned comparison
unsigned int __vsetgeu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vsetgeu4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison
unsigned int __vsetgts2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vsetgts4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison
unsigned int __vsetgtu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vsetgtu4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned minimum computation
unsigned int __vsetles2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vsetles4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison
unsigned int __vsetleu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vsetleu4(unsigned int a,unsigned int b);
//Performs per-halfword signed comparison
unsigned int __vsetlts2(unsigned int a,unsigned int b);
//Performs per-byte signed comparison
unsigned int __vsetlts4(unsigned int a,unsigned int b);
//Performs per-halfword unsigned comparison
unsigned int __vsetltu2(unsigned int a,unsigned int b);
//Performs per-byte unsigned comparison
unsigned int __vsetltu4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed comparison
unsigned int __vsetne2(unsigned int a,unsigned int b);
//Performs per-byte (un)signed comparison
unsigned int __vsetne4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed substraction, with wrap-around
unsigned int __vsub2(unsigned int a,unsigned int b);
//Performs per-byte substraction
unsigned int __vsub4(unsigned int a,unsigned int b);
//Performs per-halfword (un)signed substraction, with signed saturation
unsigned int __vsubss2(unsigned int a,unsigned int b);
//Performs per-byte substraction with signed saturation
unsigned int __vsubss4(unsigned int a,unsigned int b);
//Performs per-halfword substraction with unsigned saturation
unsigned int __vsubus2(unsigned int a,unsigned int b);
//Performs per-byte substraction with unsigned saturation
unsigned int __vsubus4(unsigned int a,unsigned int b);
//Performs half2 vector division in round-to-nearest-even mode
__half2 __h2div(const __half2 a,const __half2 b);
//Performs half addition in round-to-nearest-even mode
__half __hadd(const __half a,const __half b);
//Performs half addition in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half __hadd_sat(const __half a,const __half b);
//Performs half division in round-to-nearest-even mode
__half __hdiv(const __half a,const __half b);
//Performs half fused multiply-add in round-to-nearest-even mode
__half __hfma(const __half a,const __half b,const __half c);
//Performs half fused multiply-add in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half __hfma_sat(const __half a,const __half b,const __half c);
//Performs half multiplication in round-to-nearest-even mode
__half __hmul(const __half a,const __half b);
//Performs half multiplication in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half __hmul_sat(const __half a,const __half b);
//Negates input half number and returns the result
__half __hneg(const __half a);
//Performs half subtraction in round-to-nearest-even mode
__half __hsub(const __half a,const __half b);
//Performs half subtraction in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half __hsub_sat(const __half a,const __half b);
//Performs half2 vector addition in round-to-nearest-even mode
__half2 __hadd2(const __half2 a,const __half2 b);
//Performs half2 vector addition in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half2 __hadd2_sat(const __half2 a,const __half2 b);
//Performs half2 vector fused multiply-add in round-to-nearest-even mode
__half2 __hfma2(const __half2 a,const __half2 b,const __half2 c);
//Performs half2 vector fused multiply-add in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half2 __hfma2_sat(const __half2 a,const __half2 b,const __half2 c);
//Performs half2 vector multiplication in round-to-nearest-even mode
__half2 __hmul2(const __half2 a,const __half2 b);
//Performs half2 vector multiplication in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half2 __hmul2_sat(const __half2 a,const __half2 b);
//Negates both halves of the input half2 number and returns the result
__half2 __hneg2(const __half2 a);
//Performs half2 vector subtraction in round-to-nearest-even mode
__half2 __hsub2(const __half2 a,const __half2 b);
//Performs half2 vector subtraction in round-to-nearest-even mode, with saturation to [0.0, 1.0]
__half2 __hsub2_sat(const __half2 a,const __half2 b);
//Performs half if-equal comparison
bool __heq(const __half a,const __half b);
//Performs half unordered if-equal comparison
bool __hequ(const __half a,const __half b);
//Performs half greater-equal comparison
bool __hge(const __half a,const __half b);
//Performs half unordered greater-equal comparison
bool __hgeu(const __half a,const __half b);
//Performs half greater-than comparison
bool __hgt(const __half a,const __half b);
//Performs half unordered greater-than comparison
bool __hgtu(const __half a,const __half b);
//Checks if the input half number is infinite
int __hisinf(const __half a);
//Determine whether half argument is a NaN
bool __hisnan(const __half a);
//Performs half less-equal comparison
bool __hle(const __half a,const __half b);
//Performs half unordered less-equal comparison
bool __hleu(const __half a,const __half b);
//Performs half less-than comparison
bool __hlt(const __half a,const __half b);
//Performs half unordered less-than comparison
bool __hltu(const __half a,const __half b);
//Performs half not-equal comparison
bool __hne(const __half a,const __half b);
//Performs half unordered not-equal comparison
bool __hneu(const __half a,const __half b);
//Performs half2 vector if-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbeq2(const __half2 a,const __half2 b);
//Performs half2 vector unordered if-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbequ2(const __half2 a,const __half2 b);
//Performs half2 vector greater-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbge2(const __half2 a,const __half2 b);
//Performs half2 vector unordered greater-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbgeu2(const __half2 a,const __half2 b);
//Performs half2 vector greater-than comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbgt2(const __half2 a,const __half2 b);
//Performs half2 vector unordered greater-than comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbgtu2(const __half2 a,const __half2 b);
//Performs half2 vector less-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hble2(const __half2 a,const __half2 b);
//Performs half2 vector unordered less-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbleu2(const __half2 a,const __half2 b);
//Performs half2 vector less-than comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hblt2(const __half2 a,const __half2 b);
//Performs half2 vector unordered less-than comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbltu2(const __half2 a,const __half2 b);
//Performs half2 vector not-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbne2(const __half2 a,const __half2 b);
//Performs half2 vector unordered not-equal comparison, and returns boolean true iff both half results are true, boolean false otherwise
bool __hbneu2(const __half2 a,const __half2 b);
//Performs half2 vector if-equal comparison
__half2 __heq2(const __half2 a,const __half2 b);
//Performs half2 vector unordered if-equal comparison
__half2 __hequ2(const __half2 a,const __half2 b);
//Performs half2 vector greater-equal comparison
__half2 __hge2(const __half2 a,const __half2 b);
//Performs half2 vector unordered greater-equal comparison
__half2 __hgeu2(const __half2 a,const __half2 b);
//Performs half2 vector greater-than comparison
__half2 __hgt2(const __half2 a,const __half2 b);
//Performs half2 vector unordered greater-than comparison
__half2 __hgtu2(const __half2 a,const __half2 b);
//Determine whether half2 argument is a NaN
__half2 __hisnan2(const __half2 a);
//Performs half2 vector less-equal comparison
__half2 __hle2(const __half2 a,const __half2 b);
//Performs half2 vector unordered less-equal comparison
__half2 __hleu2(const __half2 a,const __half2 b);
//Performs half2 vector less-than comparison
__half2 __hlt2(const __half2 a,const __half2 b);
//Performs half2 vector unordered less-than comparison
__half2 __hltu2(const __half2 a,const __half2 b);
//Performs half2 vector not-equal comparison
__half2 __hne2(const __half2 a,const __half2 b);
//Performs half2 vector unordered not-equal comparison
__half2 __hneu2(const __half2 a,const __half2 b);
//Converts both components of float2 number to half precision in round-to-nearest-even mode and returns half2 with converted values
__host__ __half2 __float22half2_rn(const float2 a);
//Converts float number to half precision in round-to-nearest-even mode and returns half with converted value
__host__ __half __float2half(const float a);
//Converts input to half precision in round-to-nearest-even mode and populates both halves of half2 with converted value
__host__ __half2 __float2half2_rn(const float a);
//Converts float number to half precision in round-down mode and returns half with converted value
__host__ __half __float2half_rd(const float a);
//Converts float number to half precision in round-to-nearest-even mode and returns half with converted value
__host__ __half __float2half_rn(const float a);
//Converts float number to half precision in round-up mode and returns half with converted value
__host__ __half __float2half_ru(const float a);
//Converts float number to half precision in round-towards-zero mode and returns half with converted value
__host__ __half __float2half_rz(const float a);
//Converts both input floats to half precision in round-to-nearest-even mode and returns half2 with converted values
__host__ __half2 __floats2half2_rn(const float a,const float b);
//Converts both halves of half2 to float2 and returns the result
__host__ float2 __half22float2(const __half2 a);
//Converts half number to float
__host__ float __half2float(const __half a);
//Returns half2 with both halves equal to the input value
__half2 __half2half2(const __half a);
//Convert a half to a signed integer in round-down mode
int __half2int_rd(__half h);
//Convert a half to a signed integer in round-to-nearest-even mode
int __half2int_rn(__half h);
//Convert a half to a signed integer in round-up mode
int __half2int_ru(__half h);
//Convert a half to a signed integer in round-towards-zero mode
int __half2int_rz(__half h);
//Convert a half to a signed 64-bit integer in round-down mode
long long int __half2ll_rd(__half h);
//Convert a half to a signed 64-bit integer in round-to-nearest-even mode
long long int __half2ll_rn(__half h);
//Convert a half to a signed 64-bit integer in round-up mode
long long int __half2ll_ru(__half h);
//Convert a half to a signed 64-bit integer in round-towards-zero mode
long long int __half2ll_rz(__half h);
//Convert a half to a signed short integer in round-down mode
short int __half2short_rd(__half h);
//Convert a half to a signed short integer in round-to-nearest-even mode
short int __half2short_rn(__half h);
//Convert a half to a signed short integer in round-up mode
short int __half2short_ru(__half h);
//Convert a half to a signed short integer in round-towards-zero mode
short int __half2short_rz(__half h);
//Convert a half to an unsigned integer in round-down mode
unsigned int __half2uint_rd(__half h);
//Convert a half to an unsigned integer in round-to-nearest-even mode
unsigned int __half2uint_rn(__half h);
//Convert a half to an unsigned integer in round-up mode
unsigned int __half2uint_ru(__half h);
//Convert a half to an unsigned integer in round-towards-zero mode
unsigned int __half2uint_rz(__half h);
//Convert a half to an unsigned 64-bit integer in round-down mode
unsigned long long int __half2ull_rd(__half h);
//Convert a half to an unsigned 64-bit integer in round-to-nearest-even mode
unsigned long long int __half2ull_rn(__half h);
//Convert a half to an unsigned 64-bit integer in round-up mode
unsigned long long int __half2ull_ru(__half h);
//Convert a half to an unsigned 64-bit integer in round-towards-zero mode
unsigned long long int __half2ull_rz(__half h);
//Convert a half to an unsigned short integer in round-down mode
unsigned short int __half2ushort_rd(__half h);
//Convert a half to an unsigned short integer in round-to-nearest-even mode
unsigned short int __half2ushort_rn(__half h);
//Convert a half to an unsigned short integer in round-up mode
unsigned short int __half2ushort_ru(__half h);
//Convert a half to an unsigned short integer in round-towards-zero mode
unsigned short int __half2ushort_rz(__half h);
//Reinterprets bits in a half as a signed short integer
short int __half_as_short(const __half h);
//Reinterprets bits in a half as an unsigned short integer
unsigned short int __half_as_ushort(const __half h);
//Combines two half numbers into one half2 number
__half2 __halves2half2(const __half a,const __half b);
//Converts high 16 bits of half2 to float and returns the result
__host__ float __high2float(const __half2 a);
//Returns high 16 bits of half2 input
__half __high2half(const __half2 a);
//Extracts high 16 bits from half2 input
__half2 __high2half2(const __half2 a);
//Extracts high 16 bits from each of the two half2 inputs and combines into one half2 number
__half2 __highs2half2(const __half2 a,const __half2 b);
//Convert a signed integer to a half in round-down mode
__half __int2half_rd(int i);
//Convert a signed integer to a half in round-to-nearest-even mode
__half __int2half_rn(int i);
//Convert a signed integer to a half in round-up mode
__half __int2half_ru(int i);
//Convert a signed integer to a half in round-towards-zero mode
__half __int2half_rz(int i);
//Convert a signed 64-bit integer to a half in round-down mode
__half __ll2half_rd(long long int i);
//Convert a signed 64-bit integer to a half in round-to-nearest-even mode
__half __ll2half_rn(long long int i);
//Convert a signed 64-bit integer to a half in round-up mode
__half __ll2half_ru(long long int i);
//Convert a signed 64-bit integer to a half in round-towards-zero mode
__half __ll2half_rz(long long int i);
//Converts low 16 bits of half2 to float and returns the result
__host__ float __low2float(const __half2 a);
//Returns low 16 bits of half2 input
__half __low2half(const __half2 a);
//Extracts low 16 bits from half2 input
__half2 __low2half2(const __half2 a);
//Swaps both halves of the half2 input
__half2 __lowhigh2highlow(const __half2 a);
//Extracts low 16 bits from each of the two half2 inputs and combines into one half2 number
__half2 __lows2half2(const __half2 a,const __half2 b);
//Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller
__half __shfl_down_sync(unsigned mask,__half var,unsigned int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller
__half2 __shfl_down_sync(unsigned mask,__half2 var,unsigned int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Direct copy from indexed thread
__half __shfl_sync(unsigned mask,__half var,int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Direct copy from indexed thread
__half2 __shfl_sync(unsigned mask,__half2 var,int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller
__half __shfl_up_sync(unsigned mask,__half var,unsigned int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller
__half2 __shfl_up_sync(unsigned mask,__half2 var,unsigned int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID
__half __shfl_xor_sync(unsigned mask,__half var,int delta,int width=warpSize);
//Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID
__half2 __shfl_xor_sync(unsigned mask,__half2 var,int delta,int width=warpSize);
//Convert a signed short integer to a half in round-down mode
__half __short2half_rd(short int i);
//Convert a signed short integer to a half in round-to-nearest-even mode
__half __short2half_rn(short int i);
//Convert a signed short integer to a half in round-up mode
__half __short2half_ru(short int i);
//Convert a signed short integer to a half in round-towards-zero mode
__half __short2half_rz(short int i);
//Reinterprets bits in a signed short integer as a half
__half __short_as_half(const short int i);
//Convert an unsigned integer to a half in round-down mode
__half __uint2half_rd(unsigned int i);
//Convert an unsigned integer to a half in round-to-nearest-even mode
__half __uint2half_rn(unsigned int i);
//Convert an unsigned integer to a half in round-up mode
__half __uint2half_ru(unsigned int i);
//Convert an unsigned integer to a half in round-towards-zero mode
__half __uint2half_rz(unsigned int i);
//Convert an unsigned 64-bit integer to a half in round-down mode
__half __ull2half_rd(unsigned long long int i);
//Convert an unsigned 64-bit integer to a half in round-to-nearest-even mode
__half __ull2half_rn(unsigned long long int i);
//Convert an unsigned 64-bit integer to a half in round-up mode
__half __ull2half_ru(unsigned long long int i);
//Convert an unsigned 64-bit integer to a half in round-towards-zero mode
__half __ull2half_rz(unsigned long long int i);
//Convert an unsigned short integer to a half in round-down mode
__half __ushort2half_rd(unsigned short int i);
//Convert an unsigned short integer to a half in round-to-nearest-even mode
__half __ushort2half_rn(unsigned short int i);
//Convert an unsigned short integer to a half in round-up mode
__half __ushort2half_ru(unsigned short int i);
//Convert an unsigned short integer to a half in round-towards-zero mode
__half __ushort2half_rz(unsigned short int i);
//Reinterprets bits in an unsigned short integer as a half
__half __ushort_as_half(const unsigned short int i);
//Calculate ceiling of the input argument
__half hceil(const __half h);
//Calculates half cosine in round-to-nearest-even mode
__half hcos(const __half a);
//Calculates half natural exponential function in round-to-nearest mode
__half hexp(const __half a);
//Calculates half decimal exponential function in round-to-nearest mode
__half hexp10(const __half a);
//Calculates half binary exponential function in round-to-nearest mode
__half hexp2(const __half a);
//Calculate the largest integer less than or equal to h
__half hfloor(const __half h);
//Calculates half natural logarithm in round-to-nearest-even mode
__half hlog(const __half a);
//Calculates half decimal logarithm in round-to-nearest-even mode
__half hlog10(const __half a);
//Calculates half binary logarithm in round-to-nearest-even mode
__half hlog2(const __half a);
//Calculates half reciprocal in round-to-nearest-even mode
__half hrcp(const __half a);
//Round input to nearest integer value in half-precision floating point number
__half hrint(const __half h);
//Calculates half reciprocal square root in round-to-nearest-even mode
__half hrsqrt(const __half a);
//Calculates half sine in round-to-nearest-even mode
__half hsin(const __half a);
//Calculates half square root in round-to-nearest-even mode
__half hsqrt(const __half a);
//Truncate input argument to the integral part
__half htrunc(const __half h);
//Calculate half2 vector ceiling of the input argument
__half2 h2ceil(const __half2 h);
//Calculates half2 vector cosine in round-to-nearest-even mode
__half2 h2cos(const __half2 a);
//Calculates half2 vector exponential function in round-to-nearest mode
__half2 h2exp(const __half2 a);
//Calculates half2 vector decimal exponential function in round-to-nearest-even mode
__half2 h2exp10(const __half2 a);
//Calculates half2 vector binary exponential function in round-to-nearest-even mode
__half2 h2exp2(const __half2 a);
//Calculate the largest integer less than or equal to h
__half2 h2floor(const __half2 h);
//Calculates half2 vector natural logarithm in round-to-nearest-even mode
__half2 h2log(const __half2 a);
//Calculates half2 vector decimal logarithm in round-to-nearest-even mode
__half2 h2log10(const __half2 a);
//Calculates half2 vector binary logarithm in round-to-nearest-even mode
__half2 h2log2(const __half2 a);
//Calculates half2 vector reciprocal in round-to-nearest-even mode
__half2 h2rcp(const __half2 a);
//Round input to nearest integer value in half-precision floating point number
__half2 h2rint(const __half2 h);
//Calculates half2 vector reciprocal square root in round-to-nearest mode
__half2 h2rsqrt(const __half2 a);
//Calculates half2 vector sine in round-to-nearest-even mode
__half2 h2sin(const __half2 a);
//Calculates half2 vector square root in round-to-nearest-even mode
__half2 h2sqrt(const __half2 a);
//Truncate half2 vector input argument to the integral part
__half2 h2trunc(const __half2 h);
//Evaluate predicate for all non-exited threads in mask, return non-zero if predicate is non-zero for all of them
int __all_sync(unsigned mask,int predicate);
//Evaluate predicate for all non-exited threads in mask, return non-zero if predicate is non-zero for any of them
int __any_sync(unsigned mask,int predicate);
//Evaluate predicate for all non-exited threads in mask, return integer with
//Nth bit set if predicate is non-zero for Nth thread of warp when Nth thread is active
unsigned __ballot_sync(unsigned mask,int predicate);
//Returns a 32-bit mask of active threads in calling warp. Nth bit is set if Nth lange in warp is active
//Inactive threads are 0 bits in returned mask. Exited threads are marked as inactive. Does not synchronize following functions
unsigned __activemask();
//Broadcast-and-compare - returns mask of threads that have the same value in mask(CC 7.x+)
template<typename T>unsigned int __match_any_sync(unsigned mask,T value);
//Broadcast-and-compare - returns mask if all threads in mask have the same value, otherwise 0
//Pred is set to true if all threads in mask have the same value, otherwise false(CC 7.x+)
template<typename T>unsigned int __match_all_sync(unsigned mask,T value,int*pred);
//Direct copy from indexed lane
//Returns the value of var held by the thread whose ID is given by srcLane
//If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0
//If srcLane is outside the range [0:width-1], the value returned corresponds to the value of var held by the srcLane modulo width
template<typename T>T __shfl_sync(unsigned mask,T var,int srcLane,int width=warpSize);
//Copy from a lane with lower ID relative to called
//Calculates a source lane ID by subtracting delta from the caller's lane ID
//The value of var held by the resulting lane ID is returned: in effect, var is shifted up the warp by delta lanes
//If width is less than warpSize then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0
//The source lane index will not wrap around the value of width, so effectively the lower delta lanes will be unchanged
template<typename T>T __shfl_up_sync(unsigned mask,T var,unsigned int delta,int width=warpSize);
//Copy from a lane with higher ID relative to called
//Same as __shfl_up_sync for higher ID
template<typename T>T __shfl_down_sync(unsigned mask,T var,unsigned int delta,int width=warpSize);
//Copy from a lane based on bitwise XOR of own lane ID
//Calculates a source line ID by performing a bitwise XOR of the caller's lane ID with laneMask: the value of var held by the resulting lane ID is returned
//If width is less than warpSize then each group of width consecutive threads are able to access elements from earlier groups of threads
//however if they attempt to access elements from later groups of threads their own value of var will be returned
//This mode implements a butterfly addressing pattern such as is used in tree reduction and broadcast
template<typename T> T __shfl_xor_sync(unsigned mask,T var,int laneMask,int width=warpSize);
#endif
