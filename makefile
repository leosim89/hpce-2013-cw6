SHELL=/bin/bash

CC=g++-4.7
CPPFLAGS += -std=c++11 -W -Wall -g
CPPFLAGS += -O3
CPPFLAGS += -I include
LDFLAGS += -lrt -ltbb -lOpenCL

# For your makefile, add TBB and OpenCL as appropriate

# Launch client and server connected by pipes
launch_pipes : src/bitecoin_server src/bitecoin_client
	-rm .fifo_rev
	mkfifo .fifo_rev
	# One direction via pipe, other via fifo
	src/bitecoin_client client1 3 file .fifo_rev - | (src/bitecoin_server server1 3 file - .fifo_rev &> /dev/null)

# Launch an "infinite" server, that will always relaunch
launch_infinite_server : src/bitecoin_server
	while [ 1 ]; do \
		src/bitecoin_server server1-$USER 3 tcp-server 4000; \
	done;

# Launch a client connected to a local server
connect_local : src/bitecoin_client
	src/bitecoin_client CareBear 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
connect_exchange : src/bitecoin_miner
	src/bitecoin_miner client-$(USER) 3 tcp-client $(EXCHANGE_ADDR)  $(EXCHANGE_PORT)

src/bitecoin_client:
	$(CC) $(CPPFLAGS) src/bitecoin_client.cpp $(LDFLAGS) -o src/bitecoin_client

src/bitecoin_server:
	$(CC) $(CPPFLAGS) src/bitecoin_server.cpp $(LDFLAGS) -o src/bitecoin_server

src/bitecoin_miner:
	$(CC) $(CPPFLAGS) src/bitecoin_miner.cpp $(LDFLAGS) -o src/bitecoin_miner
