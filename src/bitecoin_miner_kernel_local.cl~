uint wide_add(uint n, __global uint *res, uint *a, __global const uint *b)
{
	ulong carry=0;
	for(uint i=0;i<n;i++){
		ulong tmp= convert_ulong(a[i])+convert_ulong(b[i])+carry;
		res[i]=convert_uint(tmp&0x00000000FFFFFFFF);
		carry=tmp>>32;
	}
	return carry;
}

uint wide_add_carry(uint n, __global uint *res, uint *a, uint b)
{
	ulong carry=b;
	for(uint i=0;i<n;i++){
		ulong tmp= convert_ulong(a[i]) +carry;
		res[i] = convert_uint(tmp&0x00000000FFFFFFFF);
		carry=tmp>>32;
	}
	return carry;
}

void wide_mul(uint n, uint *res_hi, uint *res_lo, __global const uint *a, __global const uint *b)
{
	ulong carry=0, acc=0;
	for(uint i=0; i<n; i++){
		for(uint j=0; j<=i; j++){
			ulong tmp = convert_ulong(a[j])*convert_ulong(b[i-j]);
			acc+=tmp;
			if(acc < tmp)
				carry++;
		}
		res_lo[i]=convert_uint(acc&0x00000000FFFFFFFF);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	
	for(uint i=1; i<n; i++){
		for(uint j=i; j<n; j++){
			ulong tmp= convert_ulong(a[j])*convert_ulong(b[n-j+i-1]);
			acc+=tmp;
			if(acc < tmp)
				carry++;
		}
		res_hi[i-1]= convert_uint(acc&0x00000000FFFFFFFF);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	res_hi[n-1] = convert_uint(acc);
}


	
__kernel void main_loop(
	uint hashSteps,
	__global uint *c,
	__global uint *indices,
	__global uint *point,
	__global const uint *temp,
	__local uint *localIndices
	){
		uint i=get_global_id(0);	// Counter for indices
		uint k=get_global_id(1);	// Counter for iterations
		const uint j = get_global_size(0);
		
		uint tmp[8];
		
		uint i_l = get_local_id(0); //iterations
		uint k_l = get_local_id(1);
		
		uint ind = i_l + k_l*j;
		
		localIndices[ind] = indices[k*j+i];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Calculate the hash for this specific point
		for (uint x = 1; x < 8; x++){
			point[(k*j+i)*8+x] = temp[x];
		}
		point[(k*j+i)*8] = localIndices[ind];
		
		// Now step forward by the number specified by the server
		for(uint y=0;y<hashSteps;y++){
			wide_mul(4, &tmp[4], &tmp[0], &point[(k*j+i)*8], c);
			uint carry=wide_add(4, &point[(k*j+i)*8], &tmp[0], &point[(k*j+i)*8+4]);
			wide_add_carry(4, &point[(k*j+i)*8+4], &tmp[4], carry);
		}
};
