#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> 
#include <csignal>

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

		void MakeBid(
			const std::shared_ptr<Packet_ServerBeginRound> roundInfo,	// Information about this particular round
			const std::shared_ptr<Packet_ServerRequestBid> request,		// The specific request we received
			double period,																			// How long this bidding period will last
			double skewEstimate,																// An estimate of the time difference between us and the server (positive -> we are ahead)
			std::vector<uint32_t> &solution,												// Our vector of indices describing the solution
			uint32_t *pProof																		// Will contain the "proof", which is just the value
		){
			double tSafetyMargin=0.5;
			double tFinish=request->timeStampReceiveBids*1e-9 + skewEstimate - tSafetyMargin;
		
			Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);
			
			std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
			bigint_t bestProof;
			wide_ones(BIGINT_WORDS, bestProof.limbs);
		
			unsigned nTrials=0;
			while(1){
				++nTrials;
			
				Log(Log_Debug, "Trial %d.", nTrials);
				std::vector<uint32_t> indices(roundInfo->maxIndices);
				uint32_t curr=0;
				for(unsigned j=0;j<indices.size();j++){
					curr=curr+1+(rand()%10);
					indices[j]=curr;
				}
			
				bigint_t proof;
				wide_zero(8, proof.limbs);
		
				for(unsigned i=0;i<indices.size();i++){
					if(i>0){
						if(indices[i-1] >= indices[i])
							throw std::invalid_argument("HashReference - Indices are not in monotonically increasing order.");
					}
			
					// Calculate the hash for this specific point
					hash::fnv<64> hasher;
					uint64_t chainHash=hasher((const char*)&roundInfo.get()->chainData[0], roundInfo.get()->chainData.size());
		
					// The value x is 8 words long (8*32 bits in total)
					// We build (MSB to LSB) as  [ chainHash ; roundSalt ; roundId ; index ]
					bigint_t point;
					wide_zero(8, point.limbs);
					wide_add(8, point.limbs, point.limbs, indices[i]);	//chosen index goes in at two low limbs
					wide_add(6, point.limbs+2, point.limbs+2, roundInfo.get()->roundId);	// Round goes in at limbs 3 and 2
					wide_add(4, point.limbs+4, point.limbs+4, roundInfo.get()->roundSalt);	// Salt goes in at limbs 5 and 4
					wide_add(2, point.limbs+6, point.limbs+6, chainHash);	// chainHash at limbs 7 and 6
		
					// Now step forward by the number specified by the server
					for(unsigned j=0;j<roundInfo.get()->hashSteps;j++){
						bigint_t tmp;
						wide_zero(8, tmp.limbs);
						wide_mul(4, tmp.limbs+4, tmp.limbs, point.limbs, roundInfo.get()->c);
						uint32_t carry=wide_add(4, point.limbs, tmp.limbs, point.limbs+4);
						wide_add(4, point.limbs+4, tmp.limbs+4, carry);
					}
			
					// Combine the hashes of the points together using xor
					wide_xor(8, proof.limbs, proof.limbs, point.limbs);
				}

				double score=wide_as_double(BIGINT_WORDS, proof.limbs);
				Log(Log_Debug, "    Score=%lg", score);
			
				if(wide_compare(BIGINT_WORDS, proof.limbs, bestProof.limbs)<0){
					double worst=pow(2.0, BIGINT_LENGTH*8);	// This is the worst possible score
					Log(Log_Verbose, "    Found new best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst/score);
					bestSolution=indices;
					bestProof=proof;
				}
			
				double t=now()*1e-9;	// Work out where we are against the deadline
				double timeBudget=tFinish-t;
				Log(Log_Debug, "Finish trial %d, time remaining =%lg seconds.", nTrials, timeBudget);
			
				if(timeBudget<=0)
					break;	// We have run out of time, send what we have
			}
		
			solution=bestSolution;
			wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);
		
			Log(Log_Verbose, "MakeBid - finish.");
		}
	}; //EndpointClient

}; // namespace bitecoin


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

