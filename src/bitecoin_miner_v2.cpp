#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> 
#include <csignal>
#include <string>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>

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
			
			// Best Score
			std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
			bigint_t bestProof;
			wide_ones(BIGINT_WORDS, bestProof.limbs);
			
			// Generation of Points for hashing
			hash::fnv<64> hasher;
			uint64_t chainHash=hasher((const char*)&roundInfo.get()->chainData[0], roundInfo.get()->chainData.size());
			bigint_t temp;
			temp.limbs[2] = roundInfo.get()->roundId&0xFFFFFFFFULL;
			temp.limbs[3] = temp.limbs[2];
			temp.limbs[4] = roundInfo.get()->roundSalt&0xFFFFFFFFULL;
			temp.limbs[5] = temp.limbs[4];
			temp.limbs[6] = chainHash&0xFFFFFFFFULL;
			temp.limbs[7] = temp.limbs[6];
			wide_zero(2, temp.limbs);
			
			unsigned int iterations = 16;	// Simultaneous Iterations
			
			// Variables
			uint32_t *indices;
			indices = new uint32_t[iterations*roundInfo->maxIndices];
			bigint_t *proof;
			proof = new bigint_t[iterations];
			double score[iterations];	
			
			unsigned nTrials=1;
			
			while(1){		// Trial Loop
			
				(Log_Debug, "Trials %d - %d.", nTrials, nTrials + iterations - 1);
				
				for(int k = 0; k < iterations; k++) {
					indices[(k*roundInfo->maxIndices)]=1+(rand()%10);
					for(unsigned i=1;i<roundInfo->maxIndices;i++){
						indices[i+(k*roundInfo->maxIndices)] = indices[i-1+(k*roundInfo->maxIndices)]+1+(rand()%10);
					}
					wide_zero(8, proof[k].limbs);
				}
				
		
				auto main_loop = [&] (
					unsigned int i, // Counter for index
					unsigned int k,	// Counter for iterations
					unsigned int j,	// Max for index
					unsigned int l, // Max for iterations
					uint32_t *indices,
					bigint_t *proof,
					bigint_t temp
					){
						bigint_t point;
						// Calculate the hash for this specific point
						temp.limbs[0] = indices[k*j+i];
						point = temp;
		
						// Now step forward by the number specified by the server
						for(unsigned j=0;j<roundInfo.get()->hashSteps;j++){
							PoolHashStep(point, roundInfo.get());
						}
			
						// Combine the hashes of the points together using xor
						wide_xor(8, proof[k].limbs, proof[k].limbs, point.limbs);
				};
				
				for (int k = 0; k < iterations; k++){
					for (int i = 0; i < roundInfo->maxIndices; i++){
						main_loop(i,k, roundInfo->maxIndices, iterations, &indices[0], &proof[0], temp);
					}
				}
				
				for (unsigned int k = 0; k < iterations; k++) {
					score[k]=wide_as_double(BIGINT_WORDS, proof[k].limbs);
					Log(Log_Debug, "    Score=%lg", score);
					if(wide_compare(BIGINT_WORDS, proof[k].limbs, bestProof.limbs)<0){
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
			solution=bestSolution;
			wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
		
			Log(Log_Verbose, "MakeBid - finish.");
			Log(Log_Info, "nTrials=%d, Trial rate=%f trials per second", nTrials, nTrials/period);
		}
	};

};

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

