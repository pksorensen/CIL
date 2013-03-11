
//#define ACTIVATION_FUNCTION(X) (1.7159f*tanh(2.0f/3.0f*X))

__kernel void
matrix_matrix_mul(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB, int hA)
{
	int tx = get_global_id(0);
	int ty = get_global_id(1);
	 
	tx = select(0,tx,tx<wB);
	ty = select(0,ty,ty<hA);

	//if (tx<wB && ty<hA)
	//{
		float value = 0;
		for (int k = 0; k < wA; ++k)
		{
			float elementA = A[ty * wA + k];
			float elementB = B[k * wB + tx];
			value += elementA * elementB;
		}
 
		// Write the matrix to device memory each 
		// thread writes one element
		C[ty * wA + tx] = ACTIVATION_FUNCTION(value);
	//}
}