void PoolHashStepCombined (bigint_t &x, const Packet_ServerBeginRound *pParams)
{
	bigint_t y;
	uint64_t carry=0, acc=0;
	uint64_t carryy =0;
	for(unsigned i=0; i<4; i++){
		for(unsigned j=0; j<=i; j++){
			assert( (j+(i-j))==i );
			uint64_t temp=uint64_t(x.limbs[j])*pParams->c[i-j];
			acc+=temp;
			if(acc < temp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,i-j);
		}
		uint64_t tmp=(acc&0xFFFFFFFFull)+x.limbs[i+4]+carryy;
		y.limbs[i]=uint32_t(tmp&0xFFFFFFFFULL);
		//fprintf(stderr, "\n  %d : %u\n", i, res_lo[i]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
		carryy=tmp>>32;
	}
	
	for(unsigned i=1; i<4; i++){
		for(unsigned j=i; j<4; j++){
			uint64_t temp=uint64_t(x.limbs[j])*pParams->c[3-j+i];
			acc+=temp;
			if(acc < temp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,n-j+i-1);
			//assert( (j+(n-j))==n+i );
		}
		uint64_t tmp=(acc&0xFFFFFFFFull)+carryy;
		y.limbs[i+3]=uint32_t(tmp&0xFFFFFFFFULL);
		//fprintf(stderr, "\n  %d : %u\n", i+n-1, res_hi[i-1]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
		carryy=tmp>>32;
	}
	y.limbs[7]=uint32_t((acc+carryy)&0xFFFFFFFFULL);
	x = y; 	
}	