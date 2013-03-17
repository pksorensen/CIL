

__kernel void m_m_mul( __global float *A,
						__global float *B,
						__global float *C,
						__local float *B_col, //same size as A_cols
						uint A_cols)
{

	int k;
	int A_rows = get_local_size(1)*get_num_groups(1);
	int A_row_idx = get_global_id(1);//get_group_id(1)*get_local_size(1)+get_local_id(1);
	int B_col_idx = get_global_id(0);//get_group_id(0);
	
	//Each work item in group copy A_rows/256 elements from B.
	for(k=get_local_id(0); k<A_cols;k+=get_local_size(0))
	{
		B_col[k] = A[A_row_idx+A_rows*k];//B[B_col_idx+k*get_num_groups(0)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//When all work items have copied do the output[r,c] value.
	float value = 0.0f;
	for(k=0;k<A_cols;++k)
	{
		//value +=A[A_row_idx*A_cols+k]*B[B_col_idx+k*get_num_groups(0)];//B_col[k];
		
		//value +=B[B_col_idx*A_cols+k]*A[A_row_idx*A_cols+k];
		//value +=B_col[k];
		//value += B[B_col_idx+k*get_num_groups(0)];
		//value +=B_col[k]*B[B_col_idx+k*get_num_groups(0)];

		value +=B[B_col_idx*A_cols+k]*B_col[k];
		//value +=B_col[k];
	}
	C[A_row_idx+A_rows*B_col_idx] = value;
	

}