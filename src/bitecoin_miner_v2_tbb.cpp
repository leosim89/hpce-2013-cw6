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
#include "tbb/task_scheduler_init.h"

#include <fstream>
#include <streambuf>

#define worst pow(2.0, BIGINT_LENGTH*8)

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
			
		void MakeBid(
			const std::shared_ptr<Packet_ServerBeginRound> roundInfo,	// Information about this particular round
			const std::shared_ptr<Packet_ServerRequestBid> request,		// The specific request we received
			double period,																			// How long this bidding period will last
			double skewEstimate,																// An estimate of the time difference between us and the server (positive -> we are ahead)
			std::vector<uint32_t> &solution,												// Our vector of indices describing the solution
			uint32_t *pProof																		// Will contain the "proof", which is just the value
		){
			// Time Related Calculations
			double tSafetyMargin=0.5;
			double tFinish=request->timeStampReceiveBids*1e-9 + skewEstimate - tSafetyMargin;
			Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);
			double Trialt = now()*1e-9;
			
			// Best Score
			std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
			bigint_t bestProof;
			wide_ones(BIGINT_WORDS, bestProof.limbs);
			
			// Generation of Points for hashing
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
			
			unsigned nTrials = 0;
			
			// Initialise TBB to use max number of cores
			tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
			fprintf(stderr, "Cores = %d\n", tbb::task_scheduler_init::default_num_threads());
			unsigned int iterations = tbb::task_scheduler_init::default_num_threads()*16;
			
			//Variables
			uint32_t indices[iterations*roundInfo->maxIndices];
			bigint_t proof[iterations];
			uint32_t point[iterations*roundInfo->maxIndices*8];
			double score[iterations];
			
			while(1){		// Trial Loop
				nTrials = nTrials + iterations;
				Log(Log_Debug, "Trials %d - %d.", nTrials - iterations +1, nTrials);

				auto main_loop = [&] (unsigned int k) {		
					indices[(k*roundInfo->maxIndices)]=1+(rand()%10);
					for(unsigned i=1;i<roundInfo->maxIndices;i++){
						indices[i+(k*roundInfo->maxIndices)] = indices[i-1+(k*roundInfo->maxIndices)]+1+(rand()%10);
					}
					wide_zero(8, proof[k].limbs);
	
					for (unsigned int i=0; i<roundInfo->maxIndices; i++){
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
					}
					for (unsigned int i = 0; i < roundInfo->maxIndices; i++){
						for(unsigned x=0;x<8;x++){
							proof[k].limbs[x] = proof[k].limbs[x]^point[(k*roundInfo->maxIndices+i)*8 + x];
						}
					}
				};
				tbb::parallel_for (0u, iterations, main_loop);
			
				for (unsigned int k = 0; k < iterations; k++) {
					score[k]=wide_as_double(BIGINT_WORDS, proof[k].limbs);
					Log(Log_Debug, "    Score=%lg", score);
					if(wide_compare(BIGINT_WORDS, proof[k].limbs, bestProof.limbs)<0){
						Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials + k, score[k], worst/score[k]);
						for(unsigned int i = 0; i < roundInfo->maxIndices; i++){
							bestSolution[i]=indices[k*roundInfo->maxIndices + i];
						}
						bestProof=proof[k];
					}
				}
			
				if (tFinish <= now()*1e-9)
					break;
			}

			Trialt = now()*1e-9 - Trialt;
			Log(Log_Info, "Trial time = %f", Trialt);
	
			solution=bestSolution;
			wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
	
			Log(Log_Verbose, "MakeBid - finish.");
			Log(Log_Info, "nTrials=%d, period=%lg, Trial rate=%f trials per second", nTrials, period, nTrials/Trialt);
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

					
