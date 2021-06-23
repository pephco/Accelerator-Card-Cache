#include <stdbool.h>
#ifndef cache
#define cache
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

//-------------------------------------------
//-----------Defining constants--------------
//-------------------------------------------

/*
* There are multiple different cache configurations supported by this application.
* The different configurations are listed in an enumerate to be used as an argument
* in the create_cache function.
*/
typedef enum CacheConfiguration_t { direct_mapped, two_way, four_way, fully_associative } config;

/*
* There are multiple different replacement policies supported by this application.
* The different replacement policies are listed in an enumerate to be used as an argument
* in the CreateCache() function.
*/
typedef enum ReplacementPolicy_t {random_RP, fifo_RP, lru_RP, mru_RP, lfu_RP, mfu_RP} policy;

/*
* A struct for extra meta data for a node is defined.
* This struct contains any application specific meta data.
* The int nodeId is an example and is a placeholder for now.
*/
typedef struct MetaData_t {
	int nodeId;
	int accessedOrder;
} MetaData_t;

/*
* A structure that represent a single cache line is defined.
* Within this structure a boolean is used to indicate if the data in that line is valid.
* A void pointer is used to store the pointer to the data in host memory as a tag. 
* This tag is used to indicate which data from the host memory is represented in the cache.
* A cl_mem  is created to point to the memory on the accelerator card.
* A pointer to the MetaData_t struct is included. In this struct extra data for a node can be stored.
*/
typedef struct CacheLine_t {
	bool valid;
	void* tag;
	struct MetaData_t* metaData;
	cl_mem deviceData;
} CacheLine_t;

/*
* A structure that represents the cache is defined.
* This structure contains all the parameters of the instantiated cache
* It also contains a double pointer pointing to the different cachelines in the different sets of the cache.
*/
typedef struct Cache_t {
	int memCopies;
	int ReadTransfers;
	int WriteTransfers;
	int tagSize;
	int dataSize;
	int indexBitMask;
	int addressBitShift;
	int numberOfLinesPerSet;
	int numberOfSets;
	enum CacheConfiguration_t config;
	CacheLine_t** cacheLine;
	int* replacementLine;
	enum ReplacementPolicy_t policy;
} Cache_t;

/*
* A struct that is given as an argument for the accelerator call.
* In this struct application specific arguments can be passed like parameters or constants.
*/
typedef struct AcceleratorParameters_t {
	int dataSize;
} AcceleratorParameters_t;

//-------------------------------------------
//------Defining interface functions---------
//-------------------------------------------
/*
* A function to instantiate a software cache within the application.
* Memory space is allocated based on the given arguments and a global variable,
* pointing to the allocated memory space, is set. 
* To create a cache the numberOfCacheLines is required, this has to be a power of 2 
* and determines the indexSize. 
* Next the data_size is required. This represents the amount of data stored at any node.
* The CreateCache() function will determine the total Cache size based on the dataSize 
* and the numberOfCacheLines. If this is larger than the defined MAX_SIZE an error will be asserted.
* The cache will create a cl_mem array the size of dataSize for each cache line.
* This means the data in a CacheLine_t represents the data of a single node.
* The tagSize is represents the number of bytes required for the tag.
* Last the config is required. This can be any configuration defined in the 
* cache_configuration enumeration.
* The function returns the pointer to the cache.
*/
struct Cache_t* CreateCache(
	int numberOfCacheLines, 
	int dataSize, 
	int tagSize, 
	enum CacheConfiguration_t config, 
	enum ReplacementPolicy_t policy);

/*
* This function transfers data from the host memory to the cache memory.
* The function checks if the data is already in cache and will only transfer when not already there.
* The function returns a cl_mem pointing to the cacheline where the data is stored.
* When CL_MEM_COPY_HOST_PTR is not provided as flag the data will always be written to the cache.
*/
cl_mem clCreateCacheBuffer(
	cl_context context, 
	cl_mem_flags flags, 
	size_t size, 
	void* hostAddress, 
	cl_int *errorcode_ret, 
	struct Cache_t* cachePtr);

/*
* This function transfers data back from the cache memory to the host memory.
* The host_address pointing to location in the host memory where the data will be stored
* and the cache_address pointing to the location in cache memory are provided.
* The data is not cleared in cache. This will only happen if location in the cache is 
* overwritten by the put() function.
* The function returns an integer to indicate if the transfer was successful.
* When the function returns '0' the data is successfully transfered to the cache.
*/
int clEnqueueReadCacheBuffer(
	cl_command_queue command_queue, 
	cl_bool blocking_read, 
	size_t offset, 
	size_t size, 
	void* hostAddress, 
	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list, 
	cl_event *event,
	struct Cache_t* cachePtr);

/*
* This functions frees the allocated memory space.
* Since only one cache can be instantiated no arguments are required.
*/
void FreeCache(
	struct Cache_t* cachePtr);

//-------------------------------------------
//-------Defining internal functions---------
//-------------------------------------------

/*
* A function to apply the indexbitmask created in the CreateCache() function to the hostAddress.
*/
static int GetIndex(
	int indexBitMask, 
	int addressBitShift, 
	void* hostAddress);

/*
* A function to find the way where the data is cached. 
* It loops through all ways in a set until it finds a line with valid data and the correct tag. 
*/
static int GetWay(
	void* hostAddress, 
	int setIndex, 
	struct Cache_t* cachePtr);

/*
* A function to determine in which way the data must be stored.
* It first checks if an empty way is available within the provided set.
* If all ways are full it overwrites a way based on the replacement policy.
*/
static int SetWay(
	int setIndex, 
	struct Cache_t* cachePtr);

#endif




