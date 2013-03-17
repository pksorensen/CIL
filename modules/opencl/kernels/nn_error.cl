


__kernel void nn_error( __global float *matrix,
						__local float* scratch,
						__global float *target 
			            __global float *reductions,
						unsigned length) {
    
	int global_index = get_global_id(0);
	float value = 0.0f;
	while (global_index < length) {
	{
		 float temp = target[idx+k]-matrix[idx + k];
		 reductions[idx+k] = temp;
		 value += temp*temp;  
	}
	reductions[idx+k]=value;


	 // Perform parallel reduction
  int local_index = get_local_id(0);
  scratch[local_index] = accumulator;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = get_local_size(0) / 2;
      offset > 0;
      offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = (mine < other) ? mine : other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }




}