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
#include <time.h>

#include "tbb/parallel_for.h"
#include "tbb/task_group.h"


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
		
		bigint_t wide_xor_array (unsigned int n, bigint_t *point) {
	
			bigint_t res;
			if (n == 2) {
				for (unsigned int i = 0; i < (unsigned int)BIGINT_WORDS ; i++){
					res.limbs[i]=point[0].limbs[i]^point[1].limbs[i];
				}
				return res;
			}
			else {
				n = n / 2;
				bigint_t a,b;
				if (n <= 4){
					a = wide_xor_array(n, &point[0]);
					b = wide_xor_array(n, &point[n]);
				} else {
					tbb::task_group group;
					auto fwd = [&] {a = wide_xor_array(n, &point[0]);};
					auto rev = [&] {b = wide_xor_array(n, &point[n]);};
					group.run(fwd);
					group.run(rev);	
					group.wait();
				}
				
				for (unsigned int i = 0; i < (unsigned int)BIGINT_WORDS ; i++){
					res.limbs[i]=a.limbs[i]^b.limbs[i];
				}
				return res;
			}
		}

		void MakeBid(
			const std::shared_ptr<Packet_ServerBeginRound> roundInfo,	// Information about this particular round
			const std::shared_ptr<Packet_ServerRequestBid> request,		// The specific request we received
			double period,																			// How long this bidding period will last
			double skewEstimate,																// An estimate of the time difference between us and the server (positive -> we are ahead)
			std::vector<uint32_t> &solution,												// Our vector of indices describing the solution
			uint32_t *pProof																		// Will contain the "proof", which is just the value
		){
			double tSafetyMargin=0.5;	// accounts for uncertainty in network conditions
			/* This is when the server has said all bids must be produced by, plus the
				adjustment for clock skew, and the safety margin
			*/
			double tFinish=request->timeStampReceiveBids*1e-9 + skewEstimate - tSafetyMargin;
		
			Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);
		
			
			
			std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
			bigint_t bestProof;
			wide_ones(BIGINT_WORDS, bestProof.limbs);
		
			hash::fnv<64> hasher;
			uint64_t chainHash=hasher((const char*)&roundInfo.get()->chainData[0], roundInfo.get()->chainData.size());
			bigint_t temp;
			temp.limbs[2] = roundInfo.get()->roundId&0xFFFFFFFFULL;
			temp.limbs[3] = temp.limbs[2];
			temp.limbs[4] = roundInfo.get()->roundSalt&0xFFFFFFFFULL;
			temp.limbs[5] = temp.limbs[4];
			temp.limbs[6] = chainHash&0xFFFFFFFFULL;
			temp.limbs[7] = temp.limbs[6];
					
			unsigned int iterations = 16;
			
			std::vector<std::vector<uint32_t>> indices(iterations, std::vector<uint32_t>(roundInfo->maxIndices));
			bigint_t proof[iterations];
			bigint_t point[iterations*roundInfo->maxIndices];
			double score[iterations];
			
			unsigned nTrials=1;
			while(1){		// Trial Loop
				(Log_Debug, "Trials %d - %d.", nTrials, (nTrials + iterations - 1));
		
				auto main_loop = [&] (unsigned int k) {
					
					uint32_t curr=0;
					for(unsigned j=0;j<indices[k].size();j++){
						curr=curr+1+(rand()%10);
						indices[k][j]=curr;
					}
			
					wide_zero(8, proof[k].limbs);

					for(unsigned i=0;i<indices[k].size();i++){

						// Calculate the hash for this specific point
						wide_zero(2, temp.limbs);
						temp.limbs[0] = indices[k][i];
						point[k*indices[k].size() + i] = temp;
					
						
						// Now step forward by the number specified by the server
						for(unsigned j=0;j<roundInfo.get()->hashSteps;j++){
							PoolHashStep(point[k*indices[k].size() + i], roundInfo.get());
						}
						
			
						// Combine the hashes of the points together using xor
						
						//wide_xor(8, proof[k].limbs, proof[k].limbs, point[k*indices[k].size() + i].limbs);
					}
					proof[k] = wide_xor_array(indices[k].size(), &point[k*indices[k].size()]);
					score[k]=wide_as_double(BIGINT_WORDS, proof[k].limbs);
					Log(Log_Debug, "    Score=%lg", score);
				};
				tbb::parallel_for (0u, iterations, main_loop);
			
				for (unsigned int k = 0; k < iterations; k++) {
					if(wide_compare(BIGINT_WORDS, proof[k].limbs, bestProof.limbs)<0){
						double worst=pow(2.0, BIGINT_LENGTH*8);	// This is the worst possible score
						Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials + k, score[k], worst/score[k]);
						bestSolution=indices[k];
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
			Log(Log_Verbose, "nTrials=%d, Trial rate=%f trials per second", nTrials, nTrials/period);
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

