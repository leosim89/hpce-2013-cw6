#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

#include <iostream>

#include <fcntl.h>
#include <unistd.h> 
#include <stdexcept>
#include <csignal>
#include <string>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <iostream>
#include <cstdlib>

#include "tbb/parallel_for.h"
#include "tbb/task_group.h"

#include <fstream>
#include <streambuf>

// Update: this doesn't work in windows - if necessary take it out. It is in
// here because some unix platforms complained if it wasn't heere.
# include <alloca.h>


// Update: Work around deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

void main_loop(
	unsigned int i, // Counter for index
	unsigned int k,	// Counter for iterations
	unsigned int j,	// Max for index
	unsigned int l, // Max for iterations
	unsigned int hashSteps,
	uint32_t *c,
	uint32_t *indices,
	uint32_t *point,
	uint32_t *temp
	){
		uint32_t tmp[8];
		// Calculate the hash for this specific point
		for (uint32_t x = 1; x < 8; x++){
			point[(k*j+i)*8+x] = temp[x];
		}
		point[(k*j+i)*8] = indices[k*j+i];

		// Now step forward by the number specified by the server
		for(unsigned y=0;y<hashSteps;y++){
			wide_mul(4, &tmp[4], &tmp[0], &point[(k*j+i)*8], c);
			uint32_t carry=wide_add(4, &point[(k*j+i)*8], &tmp[0], &point[(k*j+i)*8+4]);
			wide_add(4, &point[(k*j+i)*8+4], &tmp[4], carry);
		}

};

std::string LoadSource(const char *fileName)
{
    // Don't forget to change your_login here
    std::string baseDir="src";
    if(getenv("HPCE_CL_SRC_DIR")){
	baseDir=getenv("HPCE_CL_SRC_DIR");
    }

    std::string fullName=baseDir+"/"+fileName;

    std::ifstream src(fullName, std::ios::in | std::ios::binary);
    if(!src.is_open())
	throw std::runtime_error("LoadSource : Couldn't load cl file from '"+fullName+"'.");

    return std::string(
	(std::istreambuf_iterator<char>(src)), // Node the extra brackets.
	std::istreambuf_iterator<char>()
    );
}

int main()
{
	try{
	
		/****************** Open CL *************************/
		std::vector<cl::Platform> platforms;

		cl::Platform::get(&platforms);
		if(platforms.size()==0)
		throw std::runtime_error("No OpenCL platforms found.");

		std::cerr<<"Found "<<platforms.size()<<" platforms\n";
		for(unsigned i=0;i<platforms.size();i++){
			std::string vendor=platforms[i].getInfo<CL_PLATFORM_VENDOR>();
			std::cerr<<" Platform "<<i<<" : "<<vendor<<"\n";
		}

		int selectedPlatform=0;
		if(getenv("HPCE_SELECT_PLATFORM")){
			selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
		}
		std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
		cl::Platform platform=platforms.at(selectedPlatform);

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);	
		if(devices.size()==0){
			throw std::runtime_error("No opencl devices found.\n");
		}

		std::cerr<<"Found "<<devices.size()<<" devices\n";
		for(unsigned i=0;i<devices.size();i++){
			std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
			std::cerr<<" Device "<<i<<" : "<<name<<"\n";
		}

		int selectedDevice=0;
		if(getenv("HPCE_SELECT_DEVICE")){
			selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
		}
		std::cerr<<"Choosing device "<<selectedDevice<<"\n";
		cl::Device device=devices.at(selectedDevice);

		cl::Context context(devices);

		std::string kernelSource=LoadSource("main.cl");

		cl::Program::Sources sources;
		sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1)); // push on our single string
	
		cl::Program program(context, sources);
		try{
		    program.build(devices);
		}catch(...){
		    for(unsigned i=0;i<devices.size();i++){
			std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
			std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
		    }
		    throw;
		}
		// Actual
		unsigned int iterations = 16;
		uint32_t maxIndices = 16;
		uint32_t hashSteps = 16;
		uint32_t c[4];
		uint32_t temp[8];
		c[0]=4294964621;
		c[1]=4294967295;
		c[2]=3418534911;
		c[3]=2138916474;
		temp[0] = 0;
		temp[1] = 0;
		temp[2] = 2138916474;
		temp[3] = temp[2];
		temp[4] = 3418534911;
		temp[5] = temp[4];
		temp[6] = 4294967295;
		temp[7] = temp[6];
		uint32_t *indices;
		indices = new uint32_t[iterations*maxIndices];
		uint32_t *point;
		point = new uint32_t[iterations*maxIndices*8];
		uint32_t *point2;
		point2 = new uint32_t[iterations*maxIndices*8];
		bitecoin::bigint_t *proof;
		proof = new bitecoin::bigint_t[iterations];
		bitecoin::bigint_t *proof2;
		proof2 = new bitecoin::bigint_t[iterations];
		double score[iterations];
		double score2[iterations];
		
		//allocating GPU buffers
		cl::Buffer buffPoint(context, CL_MEM_WRITE_ONLY, maxIndices*iterations*8*4);
		cl::Buffer buffIndices(context, CL_MEM_READ_ONLY, 4*iterations*maxIndices);
		cl::Buffer buffC(context, CL_MEM_READ_ONLY, 4*4);
		cl::Buffer buffTemp(context, CL_MEM_READ_ONLY, 8*4);
		
		//finding the compiled kernel and creating an instance of it
		cl::Kernel kernel(program, "main_loop");

		//Setting the Kernel Params  --> can bring outside for loop?
		kernel.setArg(0, maxIndices);
		kernel.setArg(1, iterations);
		kernel.setArg(2, hashSteps);
		kernel.setArg(3, buffC);
		kernel.setArg(4, buffIndices);
		kernel.setArg(5, buffPoint);
		kernel.setArg(6, buffTemp);
		
		//creating command queue for single device
		cl::CommandQueue queue(context, device);

		//Setting up the iteration space
		cl::NDRange offset(0, 0);               // Always start iterations at x=0, y=0
		cl::NDRange globalSize(maxIndices, iterations);   // Global size must match the original loops
		cl::NDRange localSize=cl::NullRange;
		
		queue.enqueueWriteBuffer(buffTemp, CL_TRUE, 0, 8*4, &temp[0]);

		for(int k = 0; k < iterations; k++) {
			uint32_t curr=0;
			for(unsigned i=0;i<maxIndices;i++){
				curr=curr+1+(rand()%10);
				indices[i+(k*maxIndices)]=curr;
			}
			wide_zero(8, proof[k].limbs);
			wide_zero(8, proof2[k].limbs);
		}
	
		for (int k = 0; k < iterations; k++){
			for (int i = 0; i < maxIndices; i++){
				main_loop(i,k, maxIndices, iterations,hashSteps, &c[0], &indices[0], &point[0], &temp[0]);		
			}
		}
		
		queue.enqueueWriteBuffer(buffC, CL_TRUE, 0, 4*4, &c[0]);
		queue.enqueueWriteBuffer(buffIndices, CL_TRUE, 0, 4*iterations*maxIndices, &indices[0]);
		queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
		queue.enqueueBarrier();
		queue.enqueueReadBuffer(buffPoint, CL_TRUE, 0, maxIndices*iterations*8*4, &point2[0]);

		// Combine the hashes of the points together using xor
		for (int k = 0; k < iterations; k++){
			for (int i = 0; i < maxIndices; i++){
				for(unsigned x=0;x<8;x++){
					proof[k].limbs[x] = proof[k].limbs[x]^point[(k*maxIndices+i)*8 + x];
					proof2[k].limbs[x] = proof2[k].limbs[x]^point2[(k*maxIndices+i)*8 + x];
				}
			}
		}
		
		for (int k = 0; k < iterations; k++){
			//score[k]=wide_as_double(bitecoin::BIGINT_WORDS, proof[k].limbs);
			//score2[k]=wide_as_double(bitecoin::BIGINT_WORDS, proof2[k].limbs);
			//std::cerr << score[k] << " " << score2[k] << "\n";
			for (int x = 0; x < 8; x++) {
				std::cerr << proof[k].limbs[x] << " ";
			}
			std::cerr << "\n";
			for (int x = 0; x < 8; x++) {
				std::cerr << proof2[k].limbs[x] << " ";
			}
			std::cerr << "\n";
		}

		return 0;
	}catch(const std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<std::endl;
		//return 1;
	}
}

					
