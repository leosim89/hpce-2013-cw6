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


namespace bitecoin{

	class EndpointClientV1: public EndpointClient
	{
		public:
			EndpointClientV1(
				std::string clientId,
				std::string minerId,
				std::unique_ptr<Connection> &conn,
				std::shared_ptr<ILog> &log
			): EndpointClient (clientId, minerId, conn, log)
			{}
			
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
			
		void MakeBid(
			const std::shared_ptr<Packet_ServerBeginRound> roundInfo,	// Information about this particular round
			const std::shared_ptr<Packet_ServerRequestBid> request,		// The specific request we received
			double period,																			// How long this bidding period will last
			double skewEstimate,																// An estimate of the time difference between us and the server (positive -> we are ahead)
			std::vector<uint32_t> &solution,												// Our vector of indices describing the solution
			uint32_t *pProof																		// Will contain the "proof", which is just the value
		){
			double threshold = 10.0;
			try{

				/******************* Main Code **********************/
				double tSafetyMargin=0.5;
				double tFinish=request->timeStampReceiveBids*1e-9 + skewEstimate - tSafetyMargin;
				Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);
				
				std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
				bigint_t bestProof;
				wide_ones(BIGINT_WORDS, bestProof.limbs);
		
				hash::fnv<64> hasher;
				uint64_t chainHash=hasher((const char*)&roundInfo.get()->chainData[0], roundInfo.get()->chainData.size());
				uint32_t temp[8];
				temp[0] = 0;
				temp[1] = 0;
				temp[2] = roundInfo.get()->roundId&0xFFFFFFFFULL;
				temp[3] = temp[2];
				temp[4] = roundInfo.get()->roundSalt&0xFFFFFFFFULL;
				temp[5] = temp[4];
				temp[6] = chainHash&0xFFFFFFFFULL;
				temp[7] = temp[6];
				unsigned int iterations = 8;
				unsigned nTrials=1;
				uint32_t indices[iterations*roundInfo->maxIndices];
				bigint_t proof[iterations];
				uint32_t point[iterations*roundInfo->maxIndices*8];
				double score[iterations];
				
				if (period > threshold) {
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

					std::string kernelSource=LoadSource("bitecoin_miner_kernel.cl");

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
								
					//allocating GPU buffers
					cl::Buffer buffPoint(context, CL_MEM_WRITE_ONLY, roundInfo->maxIndices*iterations*8*4);
					cl::Buffer buffIndices(context, CL_MEM_READ_ONLY, 4*iterations*roundInfo->maxIndices);
					cl::Buffer buffC(context, CL_MEM_READ_ONLY, 4*4);
					cl::Buffer buffTemp(context, CL_MEM_READ_ONLY, 8*4);
				
					//finding the compiled kernel and creating an instance of it
					cl::Kernel kernel(program, "main_loop");

					//Setting the Kernel Params  --> can bring outside for loop?
					kernel.setArg(0, roundInfo->maxIndices);
					kernel.setArg(1, iterations);
					kernel.setArg(2, roundInfo.get()->hashSteps);
					kernel.setArg(3, buffC);
					kernel.setArg(4, buffIndices);
					kernel.setArg(5, buffPoint);
					kernel.setArg(6, buffTemp);
				
					//creating command queue for single device
					cl::CommandQueue queue(context, device);

					//Setting up the iteration space
					cl::NDRange offset(0, 0);               // Always start iterations at x=0, y=0
					cl::NDRange globalSize(roundInfo->maxIndices, iterations);   // Global size must match the original loops
					cl::NDRange localSize=cl::NullRange;
				
					queue.enqueueWriteBuffer(buffTemp, CL_TRUE, 0, 8*4, &temp[0]);
					
					
					while(1){		// Trial Loop
			
						(Log_Debug, "Trials %d - %d.", nTrials, nTrials + iterations - 1);
				
						for(int k = 0; k < iterations; k++) {
							uint32_t curr=0;
							for(unsigned i=0;i<roundInfo->maxIndices;i++){
								curr=curr+1+(rand()%10);
								indices[i+(k*roundInfo->maxIndices)]=curr;
							}
							wide_zero(8, proof[k].limbs);
						}
					
							queue.enqueueWriteBuffer(buffC, CL_TRUE, 0, 4*4, &roundInfo.get()->c[0]);
							queue.enqueueWriteBuffer(buffIndices, CL_TRUE, 0, 4*iterations*roundInfo->maxIndices, &indices[0]);
							queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
							queue.enqueueBarrier();
							queue.enqueueReadBuffer(buffPoint, CL_TRUE, 0, roundInfo->maxIndices*iterations*8*4, &point[0]);
					
						for (int k = 0; k < iterations; k++){
							for (int i = 0; i < roundInfo->maxIndices; i++){
								for(unsigned x=0;x<8;x++){
									proof[k].limbs[x] = proof[k].limbs[x]^point[(k*roundInfo->maxIndices+i)*8 + x];
								}
							}
						}
			
						for (unsigned int k = 0; k < iterations; k++) {
							score[k]=wide_as_double(BIGINT_WORDS, proof[k].limbs);
							Log(Log_Debug, "    Score=%lg", score);
							if(wide_compare(BIGINT_WORDS, proof[k].limbs, bestProof.limbs)<0){
								double worst=pow(2.0, BIGINT_LENGTH*8);	// This is the worst possible score
								Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials + k, score[k], worst/score[k]);
								for(int i = 0; i < roundInfo->maxIndices; i++){
									bestSolution[i]=indices[k*roundInfo->maxIndices + i];
								}
								bestProof=proof[k];
							}
						}
			
						nTrials = nTrials + iterations;
				
						if (tFinish <= now()*1e-9)
							break;
					
					}
				} else {
					while(1){		// Trial Loop
			
						(Log_Debug, "Trials %d - %d.", nTrials, nTrials + iterations - 1);
				
						for(int k = 0; k < iterations; k++) {
							uint32_t curr=0;
							for(unsigned i=0;i<roundInfo->maxIndices;i++){
								curr=curr+1+(rand()%10);
								indices[i+(k*roundInfo->maxIndices)]=curr;
							}
							wide_zero(8, proof[k].limbs);
						}
					
						for (int k = 0; k < iterations; k++){
							auto main_loop = [&] (unsigned int i) {
								uint32_t tmp[8];
								// Calculate the hash for this specific point
								for (uint32_t x = 1; x < 8; x++){
									point[(k*roundInfo->maxIndices+i)*8+x] = temp[x];
								}
								point[(k*roundInfo->maxIndices+i)*8] = indices[k*roundInfo->maxIndices+i];

								// Now step forward by the number specified by the server
								for(unsigned y=0;y<roundInfo.get()->hashSteps;y++){
									wide_mul(4, &tmp[4], &tmp[0], &point[(k*roundInfo->maxIndices+i)*8], roundInfo.get()->c);
									uint32_t carry=wide_add(4, &point[(k*roundInfo->maxIndices+i)*8], &tmp[0], &point[(k*roundInfo->maxIndices+i)*8+4]);
									wide_add(4, &point[(k*roundInfo->maxIndices+i)*8+4], &tmp[4], carry);
								}
							};
							//tbb::parallel_for (0u, roundInfo->maxIndices, main_loop);
							for (int i = 0; i < roundInfo->maxIndices; i++) main_loop(i);
						}
					
						for (int k = 0; k < iterations; k++){
							for (int i = 0; i < roundInfo->maxIndices; i++){
								for(unsigned x=0;x<8;x++){
									proof[k].limbs[x] = proof[k].limbs[x]^point[(k*roundInfo->maxIndices+i)*8 + x];
								}
							}
						}
			
						for (unsigned int k = 0; k < iterations; k++) {
							score[k]=wide_as_double(BIGINT_WORDS, proof[k].limbs);
							Log(Log_Debug, "    Score=%lg", score);
							if(wide_compare(BIGINT_WORDS, proof[k].limbs, bestProof.limbs)<0){
								double worst=pow(2.0, BIGINT_LENGTH*8);	// This is the worst possible score
								Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials + k, score[k], worst/score[k]);
								for(int i = 0; i < roundInfo->maxIndices; i++){
									bestSolution[i]=indices[k*roundInfo->maxIndices + i];
								}
								bestProof=proof[k];
							}
						}
			
						nTrials = nTrials + iterations;
				
						if (tFinish <= now()*1e-9)
							break;
					
					}
				
				}
			
		
				solution=bestSolution;
				wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
		
				Log(Log_Verbose, "MakeBid - finish.");
				Log(Log_Info, "nTrials=%d, Trial rate=%f trials per second", nTrials, nTrials/period);
			
			}catch(const std::exception &e){
				std::cerr<<"Caught exception : "<<e.what()<<std::endl;
				//return 1;
			}
		}
	};  // EndpointClient_V1

}; // namespace bitcoint

int main(int argc, char *argv[])
{
	if(argc<2){
		fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
		exit(1);
	}
	
	// We handle errors at the point of read/write
	signal(SIGPIPE, SIG_IGN);	// Just look at error codes
	
	try{		
		std::string clientId=argv[1];
		std::string minerId="David's Miner";
		
		// Control how much is being output.
		// Higher numbers give you more info
		int logLevel=atoi(argv[2]);
		fprintf(stderr, "LogLevel = %s -> %d\n", argv[2], logLevel);
		
		std::vector<std::string> spec;
		for(int i=3;i<argc;i++){
			spec.push_back(argv[i]);
		}
		
		
	
		std::shared_ptr<bitecoin::ILog> logDest=std::make_shared<bitecoin::LogDest>(clientId, logLevel);
		logDest->Log(bitecoin::Log_Info, "Created log.");
		
		std::unique_ptr<bitecoin::Connection> connection{bitecoin::OpenConnection(spec)};
		
		bitecoin::EndpointClientV1 endpoint(clientId, minerId, connection, logDest);
		endpoint.Run();

	}catch(std::string &msg){
		std::cerr<<"Caught error string : "<<msg<<std::endl;
		return 1;
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<std::endl;
		return 1;
	}catch(...){
		std::cerr<<"Caught unknown exception."<<std::endl;
		return 1;
	}
	
	return 0;
	
}

					
