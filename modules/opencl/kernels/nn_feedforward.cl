
//#define ACTIVATION_FUNCTION(X) (1.7159f*tanh(2.0f/3.0f*X))

__kernel void
nn_feedforward(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB, int hA
#if OUTPUT
		  ,__global float* target,
		  __global float* error,
		  __local float* scratch,
		  __global float* result
#endif		  
		  )
{
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	int k;
	 
	tx = select(0,tx,tx<wB);
	ty = select(0,ty,ty<hA);

	int oidx = ty * wB + tx;

	float value = 0;
	for ( k= 0; k < wA; ++k)
	{
		float elementA = A[ty * wA + k];
		float elementB = B[k * wB + tx];
		value += elementA * elementB;
	}
	value += B[k * wB + tx];//bias=1*bias_weight.
 
	C[oidx] = ACTIVATION_FUNCTION(value);
	
#if OUTPUT
		error[oidx] = target[oidx]-C[oidx];
//		error[oidx] *=error[oidx];
/*
		return;
		int local_index =   get_local_id(1)*wB+get_local_id(0);
		scratch[local_index] = error[oidx];

		barrier(CLK_LOCAL_MEM_FENCE);

		for(int offset = get_local_size(0)*get_local_size(1)/2;
			offset > 0;
			offset = offset/2)
		{
			if(local_index < offset)
			{
				float other = scratch[local_index + offset];
				//float mine = scratch[local_index];
				scratch[local_index] += other; 
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if (local_index == 0) {
			result[get_group_id(0)] = scratch[0];
		}
		*/
#endif
}