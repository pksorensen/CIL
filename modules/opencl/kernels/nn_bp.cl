

__kernel void nn_error( __global float *activation,
						__global float *error,
						__global float *out
{
	int idx = get_global_id(0);
	out[idx] = - error[idx] * (activation[idx] * (1 - activation[idx]));
}