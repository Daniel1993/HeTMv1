#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>
#include <fstream>

#include "MurmurHash2.h"

#define hashNum 1	//how many accounts shared 1 lock
#define arraySize  2621440 //total accounts number 10M = 2621440 integer
//#define threadNum 128	//threads number
#define resetNum 0	//how many threads do all write
#define totalNum 0	//how many threads do all read
//#define blockNum 5	//block number
#define bankNum 2	//transfer money between 2 accounts
#define TransEachThread 50 //how many transactions each thread do
#define iterations 50  //loop how many times

//Support for the lazylog implementation
#define lazyLogEn 0		//Set to 1 to enable lazy logging
#define lazyLogBlock (TransEachThread * bankNum)
#define lazyLogSize (blockNum * threadNum * lazyLogBlock)	//size of the lazy lock

//Support for the bloomfilter implementation
#define bloomEn 0		//Set to 1 to enable bloomfilter
#define bloomEntries 1000	//number of entries for the log
#define bloomK 5 //Number of hashfunctions
#define bloomSize (blockNum * threadNum * bloomEntries)	//size of the lazy lock

//For testing purposses
#define firstXEn 0	//Set to 1 to force transactions to happen betwen the firstX accounts
#define firstX 200	//Used by firstXEn

//versions in global memory	1,111(version),111,1(owner threadID)1(LOCKED)1(pre-locked)
/*#define getVersion(x) ((x)/1000000)
#define checkPrelock(x) ((x)%10)
#define checkLock(x) (((x)%100)/10)
#define getOwner(x) (((x)%1000000)/100)*/
#define getVersion(x) 	( ((x) >> 21) & 0x7ff)
#define checkPrelock(x) ( (x) & 0b1)
#define checkLock(x) 	( ((x) >> 1) & 0b1)
#define getOwner(x) 	( ((x) >> 2) & 0x7ffff)
#define maskVersion		0xfffe000
#define finalIdx (threadIdx.x+blockIdx.x*blockDim.x)

using namespace std;

int blockNum = 40;
int threadNum = 256;

// Simple Hash function used for bloom filters
__device__ unsigned int murmurhashCuda(const void * key, int len, unsigned int seed)
{
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.

	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	// Initialize the hash to a 'random' value

	unsigned int h = seed ^ len;

	// Mix 4 bytes at a time into the hash

	const unsigned char * data = (const unsigned char *)key;

	while (len >= 4)
	{
		unsigned int k = *(unsigned int *)data;

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;

		data += 4;
		len -= 4;
	}

	// Handle the last few bytes of the input array

	switch (len)
	{
	case 3: h ^= data[2] << 16;
	case 2: h ^= data[1] << 8;
	case 1: h ^= data[0];
		h *= m;
	};

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}


__global__ void setup_kernel(curandState *state)	//to setup random seed
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	/* Each thread gets same seed, a different sequence
	number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

__device__ int generate_kernel(curandState *state,	//random function in GPU
	int n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	int x = 0;
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];

	for (int i = 0; i < n; i++) {
		x = curand(&localState);

		if (x >0) {
			break;
		}
	}
	//printf("threadID = %d, Random result: %d\n",id,x);

	state[id] = localState;
	return x;
}

// random Function random several different numbers and store them into idx(Local array which stores idx of every source in Global memory).
__device__ void random_Kernel(int * idx, curandState* state, int size)
{
	for (int i = 0; i<bankNum; i++)
	{
		int m = 0;
		while (m<1){
			idx[i] = generate_kernel(state, 100) % size;
#if firstXEn==1
			idx[i] = generate_kernel(state, 100) % firstX;
#endif
			bool hasEqual = 0;
			for (int j = 0; j<i; j++)	//random different numbers
			{
				if (idx[i] == idx[j])
				{
					hasEqual = 1;
					break;
				}
			}
			if (hasEqual != 1)	//make sure they are different
				m++;
		}
	}
	/*idx[0] = generate_kernel(state,100)%size;
	for (int i = 0; i < bankNum; i++)
	{
	idx[i] = (idx[0]+i)%size;
	}*/
}

//openRead Function reads data from global memory to local memory. r_idx stores value and rv_idx stores version.
__device__ bool openRead_Kernel(volatile int* a //values in global memory
	, volatile int* mymutex	//versions in global memory	1,111(version),111,1(owner threadID)1(LOCKED)1(pre-locked)
	, int *rd_idx	// read set for address
	, int *r_idx	//read set for values
	, int *rv_idx	//read set for versions
	, int &rd_idx_size)//read set size
{
	if (finalIdx < totalNum + 1){
		int temp = *(mymutex + rd_idx[rd_idx_size] / hashNum);	//read version,owner,and lock from global memory
		if (checkPrelock(temp) == 0)	//check if it is locked by another thread
		{
			rv_idx[rd_idx_size] = getVersion(temp);	//read version from global memory to thread local memory
			r_idx[rd_idx_size] = a[rd_idx[rd_idx_size]];	//read data from global memory to thread local memory
		}
		else{
			return false;	//if it is locked by another thread which means another thread is modifying it, then return false
		}
		rd_idx_size++;
		return true;	//if successfully read, then return true
	}
	else
	{
		for (int j = 0; j <rd_idx_size; j++)
		{
			if (rd_idx[j] == rd_idx[rd_idx_size])
			{
				int temp = *(mymutex + rd_idx[rd_idx_size] / hashNum);
				if (getVersion(temp) == rv_idx[j])
				{
					r_idx[rd_idx_size] = a[rd_idx[rd_idx_size]];	//read data from global memory to thread local memory
					__threadfence();
					rv_idx[rd_idx_size] = getVersion(temp);	//read version from global memory to thread local memory
					rd_idx_size++;
					return true;
				}
				else
				{
					return false;
				}
			}
		}

		int temp = *(mymutex + rd_idx[rd_idx_size] / hashNum);	//read version,owner,and lock from global memory
		if (checkPrelock(temp) == 0)	//check if it is locked by another thread
		{
			rv_idx[rd_idx_size] = getVersion(temp);	//read version from global memory to thread local memory
			r_idx[rd_idx_size] = a[rd_idx[rd_idx_size]];	//read data from global memory to thread local memory
		}
		else{
			return false;	//if it is locked by another thread which means another thread is modifying it, then return false
		}
		rd_idx_size++;
		return true;	//if successfully read, then return true
	}
}

//openWrite Function do calculations and wirte result to w_idx. Version will increase by 1 and store to wv_idx.
__device__ bool openWrite_Kernel(volatile int* mymutex, int* rd_idx, int* wt_idx	//write set for address
	, int* r_idx	//read set for values
	, int* w_idx	//write set for values
	, int* rv_idx	//read set for versions
	, int* lv_idx	//lock set for versions
	, int* lc_idx	//lock set for address
	, int i			//read set idx for this variable
	, int rd_idx_size	//read set size
	, int& wt_idx_size	//lock table size
	, int& lc_idx_size)	//write set size
{
	wt_idx[wt_idx_size] = rd_idx[i];	//copy index from read set to write set
	lc_idx[lc_idx_size] = rd_idx[i] / hashNum;
	if (checkPrelock(*(mymutex + lc_idx[lc_idx_size])) == 0)	//check if it is locked by another thread
	{
		for (int l = 0; l<rd_idx_size; l++)
		{
			if (rd_idx[l] == rd_idx[i])
				r_idx[l] = r_idx[i];
		}
		for (int k = 0; k<wt_idx_size; k++)
		{
			if (wt_idx[k] == wt_idx[wt_idx_size])
			{
				w_idx[k] = r_idx[i];
				return true;
			}
		}
		w_idx[wt_idx_size] = r_idx[i];	//copy value from read set to write set
		wt_idx_size++;
		for (int j = 0; j < lc_idx_size; j++)
		{
			if (lc_idx[j] == lc_idx[lc_idx_size])
				return true;
		}
		lv_idx[lc_idx_size] = rv_idx[i];	//copy version from read set to lock set
		lc_idx_size++;
		return true;	//if succeed, return true
	}
	else
		return false;	//if failed, return false
}

//validate Function try to lock all memory this thread need to write and check if any memory this thread read is changed.
__device__ bool validate_Kernel(volatile int* mymutex, int* rd_idx, int* rv_idx, int* lv_idx, int* lc_idx, int rd_idx_size, int lc_idx_size, int * logRead, int logPos){
	int vr = 0;//flag for how many values in read set is still validate
	int vw = 0;//flag for how many values in write set is still validate

	for (int i = 0; i<rd_idx_size; i++){	//check all values in read set is still validate
		if (getVersion(*(mymutex + rd_idx[i] / hashNum)) == rv_idx[i] &&
			(checkPrelock(*(mymutex + rd_idx[i] / hashNum)) == 0 || getOwner(*(mymutex + rd_idx[i] / hashNum)) == finalIdx))//check if version changed or locking by another thread
			vr++;	//if not,increase 1
		else{
			return false;	//if one value in read set is not validate, return false
		}
	}

	__threadfence();
	for (int i = 0; i<lc_idx_size; i++)	//check if this thread can lock all accounts it needs to write back
	{
		int j = 0;	//flag of if this thread can lock one account in write set
		while (j < 1)
		{
			int temp = *(mymutex + lc_idx[i]);	//get lock,version,and owner from global memory
			if (checkLock(temp) == 1 || (checkPrelock(temp) == 1 && getOwner(temp) < finalIdx) || getVersion(temp) != lv_idx[i])	//check if version changed or locked by higher priority thread
			{
				for (int k = 0; k<i; k++)// if one of accounts is locked by a higher priority thread or version changed, unlock all accounts it already locked
				{
					int tem = *(mymutex + lc_idx[k]);	//local Lock value
					if (checkPrelock(tem) == 1 && getOwner(tem) == finalIdx)	// check if this account is still locked by itself
						//atomicCAS((int*)mymutex + lc_idx[k], tem, (tem / 1000000) * 1000000);	//keep version steady, and change it to unlock
						atomicCAS((int*)mymutex + lc_idx[k], tem, tem & maskVersion  );															//  MUDEI AQUI
				}
				return false;	//if one of accounts is locked by a higher priority thread or version changed, return false
			}
			//int localLock = lv_idx[i] * 1000000 + (finalIdx)* 100 + 1;	//local Lock value
			int localLock = (lv_idx[i] << 21) + (finalIdx << 2 ) + 1;	//									MUDEI AQUI
			if (atomicCAS((int*)mymutex + lc_idx[i], temp, localLock) == temp)	//atomic lock that account
				j++;	//if succeed, exit the loop
		}
	}


	__threadfence();
	for (int i = 0; i<lc_idx_size; i++)	// if this thread can pre-lock all accounts it needs to, really lock them
	{
		int temp2 = *(mymutex + lc_idx[i]);	//get lock,owner,and version from global memory
		if (getOwner(temp2) == finalIdx)	// if it is still locked by itself (check lock flag and owner position)
		{

			//int finallock = lv_idx[i] * 1000000 + finalIdx * 100 + 11;	// temp final lock
			int finallock = (lv_idx[i] << 21) + (finalIdx << 2 ) + 0b11;	//									MUDEI AQUI
			if (atomicCAS((int*)mymutex + lc_idx[i], temp2, finallock) == temp2)// lock one account to final lock
			{
				vw++;	//if succeed, vw++
#if lazyLogEn==1
				logRead[logPos + i] = rd_idx[i];
#endif
			}
			else
			{
				for (int j = 0; j< lc_idx_size; j++)	//if failed, which means this account lock is stealed by a higher priority thread, then unlock all accounts it has final locked
				{
					int temp3 = *(mymutex + lc_idx[j]);
					if (getOwner(temp3) == finalIdx)
						//atomicCAS((int*)mymutex + lc_idx[j], temp3, (temp3 / 1000000) * 1000000);	//change it to unlock
						atomicCAS((int*)mymutex + lc_idx[j], temp3, temp3 & maskVersion);	//			MUDEI AQUI
				}
				return false;
			}
		}
		else{
			for (int j = 0; j < lc_idx_size; j++)	//if this thread cannot final lock all accounts it needs to, unlock all accounts it has final locked
			{
				int temp3 = *(mymutex + lc_idx[j]);
				if (getOwner(temp3) == finalIdx)
					//atomicCAS((int*)mymutex + lc_idx[j], temp3, (temp3 / 1000000) * 1000000);	//change it to unlock
					atomicCAS((int*)mymutex + lc_idx[j], temp3, temp3 & maskVersion);	//			MUDEI AQUI
			}
			return false;
		}
	}
	if (vw == lc_idx_size && vr == rd_idx_size)	//check if this thread locked all accounts it needs to write back and all values in read set if validate
	{
		return true;	//if yes, return true
	}
	else//if no, unlock all accounts it has locked and return false
	{
		printf("========================Should never happen========================,vw = %d, vr = %d\n", vw, vr);
		for (int i = 0; i<lc_idx_size; i++)	//unlock all accounts it has locked
		{
			int temp3 = *(mymutex + lc_idx[i]);

			if (getOwner(temp3) == finalIdx)
			{
				//atomicCAS((int*)mymutex + lc_idx[i], temp3, (temp3 / 1000000) * 1000000);	//change it to unlock
				atomicCAS((int*)mymutex + lc_idx[i], temp3, temp3 & maskVersion);	//			MUDEI AQUI
			}
		}
		return false;
	}
}

//commit Fucntion copies results(both value and version) from local memory(write set) to global memory(data and lock).
__device__ void commit_Kernel(volatile int* a, int* wt_idx, volatile int* mymutex, int* w_idx, int* lv_idx, int* lc_idx, int wt_idx_size, int lc_idx_size, int * logWrite, int logPos)
{
	int finallock;
	for (int i = 0; i<wt_idx_size; i++)	//write all values in write set back to global memory and increase version
	{
		a[wt_idx[i]] = w_idx[i];	//write back values
#if lazyLogEn==1
		logWrite[logPos + i] = wt_idx[i];
#endif
	}
	__threadfence();
	int result;
	for (int i = 0; i<lc_idx_size; i++)
	{
		finallock = (lv_idx[i] << 21) + (finalIdx << 2) + 0b11;
		if (lv_idx[i] < 2048){	//increase version(if version is going to overflow,change it to 0)
			if ((result = atomicCAS((int*)mymutex + lc_idx[i], finallock, ((lv_idx[i] + 1) << 21) )) != finallock)
				printf("BUG! i = %d, threadIDX = %d, result = %d\n", i, finalIdx, result);
		}
		else{
			if ((result = atomicCAS((int*)mymutex + lc_idx[i], finallock, 0)) != finallock)
				printf("BUG! threadIDX = %d, result = %d\n", finalIdx, result);
		}
	}
}

/*
* Isto e' o que interessa
*/
__global__ void addKernelTransaction2(volatile int* a,/*values in global memory*/
	volatile int* mymutex,/*store lock,version,owner in format version*10000+owner*10+lock*/
	int* dev_abortcount,/*record how many times aborted*/
	curandState* state,/*random seed*/
	int size,
	int * devLogR,
	int * devLogW)
{


	// structure of read set and write set. Both of them have idx(d_idx),value(r_idx,w_idx) and version(rv_idx,wv_idx). And they are in local memory.
	int rd_idx[bankNum];		//address in read set
	int wt_idx[bankNum];		//address in write set

	int r_idx[bankNum];		//value in read set
	int rv_idx[bankNum];	//version in read set
	int w_idx[bankNum];		//value in write set
	int lv_idx[bankNum];	//version in lock set
	int lc_idx[bankNum];	//lock set for address
	int i = 0;	//how many transactions one thread need to commit

	unsigned int logOffset = 0, trOffset = 0;

#if lazyLogEn ==1
	logOffset = blockIdx.x*blockDim.x*lazyLogBlock + threadIdx.x*lazyLogBlock;
#endif
#if bloomEn ==1
	int seedVec[13] = { 0x9747b28c, 0xcc9e2d51, 0x1b873593, 0x85ebca6b, 0x452996ca, 0x029b5612, \
					0xfec76013, 0xbd21031b, 0x75a8ef9e, 0x8f4bc28d, 0xcf261d04 }; //Vector of seed values for murmurhash
	logOffset = blockIdx.x*blockDim.x*bloomEntries + threadIdx.x*bloomEntries;
#endif

	random_Kernel(rd_idx, state, size);	//get random index

	while (i < TransEachThread){	//each thread need to commit x transactions
		int rd_idx_size = 0;	//read set size
		int wt_idx_size = 0;	//write set size1
		int lc_idx_size = 0;

		bool flag = true;	//flag for whether succeessfully called one read or write or commit kernel
		bool flag2 = true;	//flag for whether succeessfully called all read or write or commit kernel


		for (int j = 0; j<bankNum; j++)	//open read all accounts from global memory
		{
			flag = openRead_Kernel(a, mymutex, rd_idx, r_idx, rv_idx, rd_idx_size); //read data and version from global memory to read set
			if (flag == false){	//if one read failed, increase abort time, change flag2 to false and exit this loop
				atomicAdd(dev_abortcount, 1);	//record abort times
				//printf("Open Read Aborted,threadid = %d,\n",finalidx);
				flag2 = false;
				break;
			}

		}
		if (flag2 == false)	// if one read failed, abort and try again
		{
			continue;
		}



		for (int j = 0; j < bankNum / 2; j++)
		{
			r_idx[rd_idx_size - 2] -= 1;	//-10 from one account
			flag = openWrite_Kernel(mymutex, rd_idx, wt_idx, r_idx, w_idx, rv_idx, lv_idx, lc_idx, rd_idx_size - j * 2 - 2, rd_idx_size, wt_idx_size, lc_idx_size);	//write changes to write set
			if (flag == false){	//check if this open write succeed
				atomicAdd(dev_abortcount, 1);	//record abort times
				//printf("Open Write Aborted,threadid = %d,\n",finalidx);
				flag2 = false;
				break;
			}

			r_idx[rd_idx_size - 1] += 1;		//+10 to the other account
			flag = openWrite_Kernel(mymutex, rd_idx, wt_idx, r_idx, w_idx, rv_idx, lv_idx, lc_idx, rd_idx_size - j * 2 - 1, rd_idx_size, wt_idx_size, lc_idx_size);	//write changes to write set
			if (flag == false){	//check if this open write succeed
				atomicAdd(dev_abortcount, 1);	//record abort times
				//printf("Open Write Aborted,threadid = %d,\n",finalidx);
				flag2 = false;
				break;
			}
		}

		if (flag2 == false)
		{
			continue;
		}

		if (validate_Kernel(mymutex, rd_idx, rv_idx, lv_idx, lc_idx, rd_idx_size, lc_idx_size, devLogR, logOffset + trOffset) == false){	//validate all read data and lock write memory addresses
			atomicAdd(dev_abortcount, 1);	//record abort times
			//printf("Validate Aborted,threadid = %d\n",finalidx);
			continue;
		}
		else	//if succeessfully validated, then commit
		{

			//printf("Validate succeed, threadIDX= %d, write_idx[0] = %d, write_idx[1] = %d, times = %d, a[0] = %d, a[1] = %d, a[2] = %d, wt[0] = %d, wt[1] = %d \n",finalIdx,wt_idx[0],wt_idx[1],*dev_abortcount,a[0],a[1],a[2],w_idx[0],w_idx[1]);
			//atomicAdd(dev_abortcount,1);
			//__threadfence();
			commit_Kernel(a, wt_idx, mymutex, w_idx, lv_idx, lc_idx, wt_idx_size, lc_idx_size, devLogW, logOffset + trOffset);	//commit from write set to global memory
			//i++;

			i++;
#if lazyLogEn == 1
			trOffset = i*bankNum;
#endif
#if bloomEn == 1
			unsigned int bloomPos = 0;
			unsigned int hashRes = 0;


			for (int n = 0; n < bankNum; n++) {
				for (int j = 0; j < bloomK; j++) {
					// WRITE SET
					//Executa a hash
					hashRes = murmurhashCuda(&rd_idx[n], 4, seedVec[j]);
					bloomPos = hashRes % bloomEntries;

					//Insere na posicao
					//devLogW[logOffset + bloomPos] |= 1;
					atomicCAS(&devLogW[logOffset + bloomPos], 0, 1);

					// READ SET
					//Executa a hash
					hashRes = murmurhashCuda(&wt_idx[n], 4, seedVec[j]);
					bloomPos = hashRes % bloomEntries;

					//Insere na posicao
					//devLogR[logOffset + bloomPos] |= 1;
					atomicCAS(&devLogW[logOffset + bloomPos], 0, 1);
				}
			}
#endif
		}
	}

}

/*
* Aloca a memoria na grafica, tranfere os dados, configura cache
* corre o kernel e tranfere os dados de volta.
*/
cudaError_t jobWithCuda(int *a/*,int *state*/, const unsigned int size, ofstream & ofile)
{
	int *dev_a = 0;	//values array
	int	*dev_mutex = 0;//mutex array
	int *dev_abortcount = 0;//abort time value

	int *dev_LogR = 0, *dev_LogW = 0;	//Log array

	int count = 0;	//initial abort time to 0
	int sum = 0;	//calculate sum of all accounts
	double timeForCal = 0;	//calculate average time cost

	bool err = 1, err2 = 1;	//Replacement for the goto

#if lazyLogEn==1
	int *lazyLogR = new int[lazyLogSize];
	int *lazyLogW = new int[lazyLogSize];
#endif
#if bloomEn==1
	int *bloomLogR = new int[bloomSize];
	int *bloomLogW = new int[bloomSize];
#endif

	cudaError_t cudaStatus;
	curandState *devStates;	//randseed array


	while (err) {
		err = false;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			////goto Error;
			continue;
		}

		cudaStatus = cudaMalloc((void **)&devStates, blockNum * threadNum * sizeof(curandState));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for curandState!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_a!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMalloc((void**)&dev_mutex, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_mutex!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMalloc((void**)&dev_abortcount, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for abortcount!");
			//goto Error;
			continue;
		}

#if lazyLogEn==1
		//BLOOM
		cudaStatus = cudaMalloc((void **)&dev_LogW, lazyLogSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_Log!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMalloc((void **)&dev_LogR, lazyLogSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_Log!");
			//goto Error;
			continue;
		}
#endif

#if bloomEn==1
		//BLOOM
		cudaStatus = cudaMalloc((void **)&dev_LogW, bloomSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_LogW!");
			//goto Error;
			continue;
		}

		cudaStatus = cudaMalloc((void **)&dev_LogR, bloomSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_LogR!");
			//goto Error;
			continue;
		}
#endif

		ofile << size << "	";
		StopWatchInterface *kernelTime = NULL;
		for (int i = 0; i <iterations && err2==1; i++)	//run this kernel iterations times and get average time
		{
			sum = 0;
			err2 = 0;

			for (unsigned int i = 0; i < size; i++)
			{
				sum += a[i];	//calculate sum before execution
			}
			printf("sum before = %d\n", sum);
			// Copy input vectors from host memory to GPU buffers.
			cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy to device failed for dev_a!");
				//goto Error;
				continue;
			}
			for (int j = 0; j<size; j++)
			{
				a[j] = 0;
			}

			cudaStatus = cudaMemcpy(dev_mutex, a, size * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy to device failed for dev_mutex!");
				//goto Error;
				continue;
			}

#if lazyLogEn==1

			for (int j = 0; j< lazyLogSize; j++)
			{
				lazyLogR[j] = lazyLogW[j] = 0;
			}

			cudaStatus = cudaMemcpy(dev_LogR, lazyLogR, lazyLogSize * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy to device failed for dev_LogR!");
				//goto Error;
				continue;
			}
			cudaStatus = cudaMemcpy(dev_LogW, lazyLogW, lazyLogSize * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy to device failed for dev_LogW!");
				//goto Error;
				continue;
			}
#endif

#if bloomEn==1

			for (int j = 0; j< bloomSize; j++)
			{
				bloomLogR[j] = bloomLogW[j] = 0;
		}

			cudaStatus = cudaMemcpy(dev_LogR, bloomLogR, bloomSize * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy to device failed for dev_LogR!");
				//goto Error;
				continue;
			}
			cudaStatus = cudaMemcpy(dev_LogW, bloomLogW, bloomSize * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy to device failed for dev_LogW!");
				//goto Error;
				continue;
			}
#endif


			cudaStatus = cudaMemcpy(dev_abortcount, &count, sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy to device failed for abortcount!");
				//goto Error;
				continue;
			}
			setup_kernel << <blockNum, threadNum >> >(devStates);
			cudaThreadSynchronize();	//synchronize threads

			cudaFuncSetCacheConfig(addKernelTransaction2, cudaFuncCachePreferL1);

			sdkCreateTimer(&kernelTime);	//for calculate time cost
			sdkResetTimer(&kernelTime);
			sdkStartTimer(&kernelTime);	//get start time


			addKernelTransaction2 << <blockNum, threadNum >> >(dev_a, dev_mutex, dev_abortcount, devStates, size, dev_LogR, dev_LogW);
			cudaThreadSynchronize();	//synchronize threads

			sdkStopTimer(&kernelTime);	//get stop time

			printf("Time for the kernel: %f ms\n", sdkGetTimerValue(&kernelTime));	//print out time cost
			timeForCal += sdkGetTimerValue(&kernelTime);	//for calculate average time cost
			double timeForCal2 = sdkGetTimerValue(&kernelTime);	//for calculate average time cost

			cudaStatus = cudaMemcpy(&count, dev_abortcount, sizeof(int), cudaMemcpyDeviceToHost);	//copy abort time from GPU to CPU
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for count!");
				//goto Error;
				continue;
			}
			cudaStatus = cudaMemcpy(a, dev_a, size*sizeof(int), cudaMemcpyDeviceToHost);	//copy results from GPU to CPU
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed for a!");
				//goto Error;
				continue;
			}
			/*cudaStatus = cudaMemcpy(state,dev_mutex,size*sizeof(int),cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for state!");
			//goto Error;
			continue;
			}*/

#if lazyLogEn==1
			cudaStatus = cudaMemcpy(lazyLogR, dev_LogR, lazyLogSize * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy failed for lazy_LogR!");
				//goto Error;
				continue;
			}

			cudaStatus = cudaMemcpy(lazyLogW, dev_LogW, lazyLogSize * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy failed for lazy_LogW!");
				//goto Error;
				continue;
			}

			/*for (int j = 0; j<threadNum-480; j ++)
			{
				int n = lazyLogBlock * j;
				printf("Transaction: %d\n", j );

				printf("W: ");
				for (int k = 0; k < bankNum; k++)
				{
					printf("%d ", lazyLogW[n + k]);
				}printf(" \n");

				printf("R: ");
				for (int k = 0; k < bankNum; k++)
				{
					printf("%d ", lazyLogR[n + k]);
				}printf(" \n");
			}*/
#endif

#if bloomEn==1

			/*int seedVec[13] = { 0x9747b28c, 0xcc9e2d51, 0x1b873593, 0x85ebca6b, 0x452996ca, 0x029b5612, \
				0xfec76013, 0xbd21031b, 0x75a8ef9e, 0x8f4bc28d, 0xcf261d04 }; //Vector of seed values for murmurhash
			*/
			cudaStatus = cudaMemcpy(bloomLogR, dev_LogR, bloomSize * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy failed for bloomR!");
				//goto Error;
				continue;
			}

			cudaStatus = cudaMemcpy(bloomLogW, dev_LogW, bloomSize * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cout << cudaStatus;
				fprintf(stderr, "cudaMemcpy failed for bloomW!");
				//goto Error;
				continue;
			}


			//Juncao
			int *bloomFR = new int[bloomEntries];
			int *bloomFW = new int[bloomEntries];
			for (int n = 0; n < bloomEntries; n++)
				bloomFR[n] = bloomFW[n] = 0;
			for (int j = 0; j < bloomSize; j += bloomEntries) {
				for (int n = 0; n < bloomEntries; n++) {
					bloomFR[n] |= bloomLogR[j + n];
					bloomFW[n] |= bloomLogW[j + n];
				}
			}

			/*delete[] bloomLogW;
			delete[] bloomLogR;*/


			//Testar se xBloom esta' presente
			//int xBloom, n, hits=0;
			/*for (xBloom = 100; xBloom <= 200; xBloom+=5) {

				for (int j = 0; j < bloomK; j++) {
					n = murmurhash2(&xBloom, 4, seedVec[j]) % bloomEntries;
					//printf(">>%d - %d \n", n, bloomFR[n]);
					if (bloomFR[n] == 1) hits++;
				}

				if (hits == bloomK)
					printf("Bloom Filter: %d is in the bloom filter\n", xBloom);
				else
					printf("Bloom Filter: %d NOT in the bloom filter\n", xBloom);

				hits = 0;
			}

			xBloom = 10;
			printf("%d: %o\n", xBloom, murmurhash2(&xBloom, 4, seedVec[0]) % bloomEntries);
			xBloom = 100;
			printf("%d: %o\n", xBloom, murmurhash2(&xBloom, 4, seedVec[0]) % bloomEntries);*/

			/*delete[] bloomFR;
			delete[] bloomFW;*/
#endif


			printf("abort %d times!\n", count);	//print out abort time



			sum = 0;
			for (int i = 0; i < size; i++)
			{
				sum += a[i];
				//printf("a[%d] = %d\n",i,a[i]);
				a[i] = 10;
				//state[i] = 0;
			}
			err2 = 1;
			printf("sum after = %d\n", sum);
		}
		if (!err2) continue;

		printf("Total Time = %f ms\n", timeForCal);
		printf("Average Time = %f ms\n", timeForCal / iterations);
		printf("Average Aborts = %d times\n", count / iterations);
		printf("Abort rate = %f \n", (float)count / (TransEachThread*threadNum));
		double throughput = blockNum*threadNum*TransEachThread / ((timeForCal / iterations) / 1000);
		printf("throughput is = %.0lf txs/second\n", throughput);

		// ---------------------
		FILE *stats_file_fp = fopen("stats.txt", "a");
		if (ftell(stats_file_fp) < 1) {
			fprintf(stats_file_fp, "#"
			"BLOCKS\t"
			"THREADS\t"
			"THROUGHPUT\t"
			"PROB_ABORT\n"
		);
		}
		fprintf(stats_file_fp, "%i\t", blockNum);
		fprintf(stats_file_fp, "%i\t", threadNum);
		fprintf(stats_file_fp, "%.0lf\t", throughput);
		fprintf(stats_file_fp, "%lf\n", (double)count / (double)(TransEachThread*threadNum*blockNum+count));
		fclose(stats_file_fp);
		// ---------------------

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
			continue;
		}

		float throughputEveryTime = timeForCal / iterations;
		ofile << size << "	" << threadNum*blockNum << "  " << threadNum << "  " << throughputEveryTime << "  " << throughput << " "<< count / iterations;
		ofile << endl;

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			//goto Error;
			continue;
		}

	}

//Error:
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "\nError code is: %s\n", cudaGetErrorString(cudaStatus));
	cudaFree(dev_abortcount);
	cudaFree(dev_a);
	cudaFree(dev_mutex);
	cudaFree(devStates);
#if lazyLogEn==1
	cudaFree(dev_LogR);
	cudaFree(dev_LogW);
#endif
#if bloomEn==1
	cudaFree(dev_LogR);
	cudaFree(dev_LogW);
#endif
	return cudaStatus;
}

int main(int argc, char * argv[])
{
	ofstream ofile;
	ofile.open("performanceForSoraia.txt", ios_base::app);

	if (argc >= 2) {
		blockNum = atoi(argv[1]);
		threadNum = atoi(argv[2]) % 1025;
	}

	for (int j = 0; j<1; j++)
	{
		printf("Blocks: %d, Threads: %d\n", blockNum, threadNum);

		//int size = (arraySize + 2000000 * j);
		int size = (arraySize );
		int *a = new int[size];	//accounts array
		printf("My Algorithm for %d accounts(transfer among %d accounts) with %d threads!!\n", size, bankNum, threadNum*blockNum);
		/*int *state = new int[size];	//locks array
		printf("222!!\n");*/

		for (int i = 0; i<size; i++)
		{
			a[i] = 10;	//initial accounts array
			//state[i] = 0;	//initial locks array
		}

		// Add vectors in parallel.
		cudaError_t cudaStatus = cudaSuccess;
		cudaStatus = jobWithCuda(a/*,state*/, size, ofile);

		//delete [] state;
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			ofile.close();
			//int wayOut;
			//std::cin >> wayOut;
			return 1;
		}
		//printf("max: %d\n",(numeric_limits<int>::max)());
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			ofile.close();
			int wayOut;
			std::cin >> wayOut;
			return 1;
		}
		delete[] a;
	}
	//ofile.close();

	//int wayOut;
	//std::cin >> wayOut;
	return 0;
}
