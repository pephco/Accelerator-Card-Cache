#include "math.h"
#include <stdlib.h> 
#include <stdio.h>
#include "cache.h"
#include <stdint.h>

//Global variables
bool printMemUsage = false;
bool printMemPercentage = false;

struct Cache_t* CreateCache(int numberOfCacheLines, int dataSize, int tagSize, enum CacheConfiguration_t config, enum ReplacementPolicy_t policy) {
	int numberOfSets, indexSize, numberOfLinesPerSet, addressBitShift;
	time_t t;
	srand((unsigned)time(&t));
	switch (config){
	case direct_mapped:
		numberOfSets = numberOfCacheLines;
		numberOfLinesPerSet = 1;
		indexSize = ceil(log10(numberOfCacheLines) / log10(2));
		break;
	case two_way:
		numberOfSets = (int)floor(numberOfCacheLines / 2);
		numberOfLinesPerSet = 2;
		indexSize = ceil(log10(numberOfCacheLines) / log10(2)) - 1;
		break;
	case four_way:
		numberOfSets = (int)floor(numberOfCacheLines / 4);
		numberOfLinesPerSet = 4;
		indexSize = ceil(log10(numberOfCacheLines) / log10(2)) - 2;
		break;
	case fully_associative:
		numberOfSets = 1;
		numberOfLinesPerSet = numberOfCacheLines;
		indexSize = 1;	
		break;
	default:
		break;
	}

	//Size of usable cache memory in bytes
	int cacheSize = numberOfCacheLines * dataSize;
	//Total amount of memory allocated
	uint64_t memoryAllocated = 0;
	
	//Allocate memory for the cache struct
	Cache_t* myCache=(Cache_t*)malloc(sizeof(Cache_t));
	memoryAllocated += sizeof(Cache_t);

	//Allocate memory for each set in the cache
	myCache->cacheLine =(CacheLine_t**)malloc((sizeof(CacheLine_t*) * numberOfSets));
	myCache->replacementLine = calloc(numberOfSets,sizeof(int));
	memoryAllocated += (sizeof(CacheLine_t*)+sizeof(int)) * numberOfSets;

	//Allocate memory for the cachelines in	each set
	for (int i = 0; i < numberOfSets; i++) {
		myCache->cacheLine[i] = (CacheLine_t*)malloc(sizeof(CacheLine_t) * numberOfLinesPerSet);
		memoryAllocated += sizeof(CacheLine_t) * numberOfLinesPerSet;
		//Allocate memory for the meta data on each cacheline
		for (int j = 0; j < numberOfLinesPerSet; j++) {
			//Allocate memory
			myCache->cacheLine[i][j].metaData = malloc(sizeof(MetaData_t));
			memoryAllocated += (sizeof(char) * dataSize) + sizeof(MetaData_t);
			//Initialize fields of CacheLine_t and MetaData_t
			myCache->cacheLine[i][j].valid = false;
			myCache->cacheLine[i][j].tag = NULL;
			myCache->cacheLine[i][j].metaData->nodeId = -1;
			myCache->cacheLine[i][j].metaData->accessedOrder = 0;
		}
	}

	//Get the bitshift from the data Size
	for (addressBitShift = 0; addressBitShift < dataSize; addressBitShift++) {
		if ((dataSize % (int)pow(2, addressBitShift + 1)) != 0)
			break;
	}

	//Initialize fields of Cache_t
	myCache->memCopies = 0;
	myCache->ReadTransfers = 0;
	myCache->WriteTransfers = 0;
	myCache->tagSize = tagSize;
	myCache->dataSize = dataSize;
	myCache->addressBitShift = addressBitShift;
	myCache->numberOfLinesPerSet = numberOfLinesPerSet;
	myCache->numberOfSets = numberOfSets;
	myCache->config = config;
	myCache->policy = policy;
	if (config == fully_associative)
		myCache->indexBitMask = 0;
	else
		myCache->indexBitMask = pow(2, indexSize) - 1;

	//Printf information about the memory allocation
	if (printMemUsage) {
		const char configString[4][22] = {
			"direct mapped",
			"two way associative",
			"four way associative",
			"fully associative" };
		printf("--------Memory allocation--------\n");
		printf("Cache configuration = %s\n", configString[config]);
		printf("Usable cache memory = %d bytes\n", cacheSize);
		printf("Total amount of allocated memory = %ld bytes\n", memoryAllocated);
		printf("---------------------------------\n");
	}
	if (printMemPercentage) {
		float percentage = ((float)cacheSize / (float)memoryAllocated) * 100;
		printf("Percentage of usable cache memory = %f\n",percentage);
	}
	return myCache;
}

static int GetIndex(int indexBitMask, int addressBitShift, void* hostAddress) {
	return (int)(((intptr_t)hostAddress >> addressBitShift)& indexBitMask);
}

static int GetWay( void* hostAddress, int setIndex, struct Cache_t* cachePtr) {
	int numberOfLinesPerSet = cachePtr->numberOfLinesPerSet;
	CacheLine_t* Set = cachePtr->cacheLine[setIndex];

	if(cachePtr->config==direct_mapped){
		//For direct mapped cache the way is always 0
		return 0;
	}	

	//Check for each way if the data valid and the tag is correct, if so return 0 else -1.
	for (int way = 0; way < numberOfLinesPerSet; way++) {
		if ((Set[way].valid == true) && (Set[way].tag == hostAddress)) {
			//It is in cache, update replacement policies for accessed way
			if (cachePtr->policy == lru_RP || cachePtr->policy == mru_RP) {
				cachePtr->replacementLine[setIndex]++;
				cachePtr->cacheLine[setIndex][way].metaData->accessedOrder = cachePtr->replacementLine[setIndex];
			}
			else if (cachePtr->policy == lfu_RP || cachePtr->policy == mfu_RP) {
				cachePtr->cacheLine[setIndex][way].metaData->accessedOrder++;
			}
			return way;
		}
	};
	return -1;
}

static int SetWay(int setIndex, struct Cache_t* cachePtr) {
	int numberOfLinesPerSet = cachePtr->numberOfLinesPerSet;
	CacheLine_t* Set = cachePtr->cacheLine[setIndex];
	int replacementWay = 0;
	switch (cachePtr->policy)
	{
	case random_RP:
		//Check for empty ways
		for (int way = 0; way < numberOfLinesPerSet; way++) {
			if (Set[way].valid != true)
				return way;
		};
		return rand() % numberOfLinesPerSet;
	case fifo_RP:
		cachePtr->replacementLine[setIndex] = (cachePtr->replacementLine[setIndex] + 1) % numberOfLinesPerSet;
		return cachePtr->replacementLine[setIndex];
	case lru_RP:
		for (int way = 0; way < numberOfLinesPerSet; way++) {
			if (Set[way].metaData->accessedOrder < Set[replacementWay].metaData->accessedOrder)
				replacementWay = way;
		};
		cachePtr->replacementLine[setIndex]++;
		cachePtr->cacheLine[setIndex][replacementWay].metaData->accessedOrder = cachePtr->replacementLine[setIndex];
		return replacementWay;
	case mru_RP:
		for (int way = 0; way < numberOfLinesPerSet; way++) {
			//Check for empty ways
			if (Set[way].valid != true) {
				replacementWay = way;
				break;
			}
			//Find the way with the highest accessed order
			if (Set[way].metaData->accessedOrder > Set[replacementWay].metaData->accessedOrder)
				replacementWay = way;
		};
		cachePtr->replacementLine[setIndex]++;
		cachePtr->cacheLine[setIndex][replacementWay].metaData->accessedOrder = cachePtr->replacementLine[setIndex];
		return replacementWay;
	case lfu_RP:
		for (int way = 0; way < numberOfLinesPerSet; way++) {
			if (Set[way].metaData->accessedOrder < Set[replacementWay].metaData->accessedOrder)
				replacementWay = way;
		};
		cachePtr->cacheLine[setIndex][replacementWay].metaData->accessedOrder = 1;
		return replacementWay;
	case mfu_RP:
		for (int way = 0; way < numberOfLinesPerSet; way++) {
			//Check for empty ways
			if (Set[way].valid != true) {
				replacementWay = way;
				break;
			}
			//Find the way with the highest accessed order
			if (Set[way].metaData->accessedOrder > Set[replacementWay].metaData->accessedOrder)
				replacementWay = way;
		};
		cachePtr->cacheLine[setIndex][replacementWay].metaData->accessedOrder = 1;
		return replacementWay;
	default:
		break;
	}
	//If all ways are full and no replacement policy active return 0
	return 0;
}

cl_mem clCreateCacheBuffer(cl_context context, cl_mem_flags flags, size_t size, void* hostAddress, cl_int *errorcode_ret, struct Cache_t* cachePtr){
	int set = GetIndex(cachePtr->indexBitMask, cachePtr->addressBitShift, hostAddress);
	int way = GetWay(hostAddress, set, cachePtr);

	if (!(((flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR) && (cachePtr->cacheLine[set][way].valid == true) && (cachePtr->cacheLine[set][way].tag == hostAddress))){
		//Data is not in cache or is forced to update

		//Find way to store data
		if (way == -1) 
			way = SetWay(set, cachePtr);

		if(hostAddress != NULL){
			cachePtr->cacheLine[set][way].deviceData = clCreateBuffer(context,  flags | CL_MEM_COPY_HOST_PTR, cachePtr->dataSize, hostAddress, errorcode_ret);
		}else if (hostAddress == NULL){
			cachePtr->cacheLine[set][way].deviceData = clCreateBuffer(context,  flags, cachePtr->dataSize, hostAddress, errorcode_ret);
		}
		cachePtr->memCopies += 1;
		if((flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR)
			cachePtr->WriteTransfers += 1;

		//Set the tag
		cachePtr->cacheLine[set][way].tag = hostAddress;

		//Set cacheline to valid
		cachePtr->cacheLine[set][way].valid = true;
	}
	return cachePtr->cacheLine[set][way].deviceData;
}


int clEnqueueReadCacheBuffer(cl_command_queue command_queue, cl_bool blocking_read, size_t offset, size_t size, void* hostAddress, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, struct Cache_t* cachePtr){
	//Get location in Cache
	int set = GetIndex(cachePtr->indexBitMask, cachePtr->addressBitShift, hostAddress);
	int way = GetWay(hostAddress, set, cachePtr);
	cl_int err = 0;

	if ((cachePtr->cacheLine[set][way].valid == true) && (cachePtr->cacheLine[set][way].tag == hostAddress)) {
   	err = clEnqueueReadBuffer(command_queue, cachePtr->cacheLine[set][way].deviceData , blocking_read, offset, cachePtr->dataSize, hostAddress, num_events_in_wait_list, event_wait_list, event);  
		cachePtr->memCopies += 1;
		cachePtr->ReadTransfers += 1;
		return 0;
	}
	return 1;
}

void FreeCache(struct Cache_t* cachePtr) {
	int numberOfLinesPerSet = cachePtr->numberOfLinesPerSet;
	int numberOfSets = cachePtr->numberOfSets;
	
	for (int i = 0; i < numberOfSets; i++) {
		for (int j = 0; j < numberOfLinesPerSet; j++) {
			//Free the data and meta data from the cachelines
			free(cachePtr->cacheLine[i][j].metaData);
			clReleaseMemObject(cachePtr->cacheLine[i][j].deviceData);
		}	
		//Free the cachelines from the sets
		free(cachePtr->cacheLine[i]);
	}
	//Free the sets from the cache
	free(cachePtr->replacementLine);
	free(cachePtr->cacheLine);
	//Free the cache
	free(cachePtr);
}

