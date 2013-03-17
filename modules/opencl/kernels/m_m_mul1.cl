#define BLOCK_SIZE 16
//__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
__kernel  void
m_m_mul(__global       float * M1,
        __global       float * M2,
        __global       float * M3,
                             uint qq)
{
   //Identification of this workgroup
   int i = get_group_id(0);
   int j = get_group_id(1);

   //Identification of work-item
   int idX = get_local_id(0); 
   int idY = get_local_id(1);

   //matrixes dimensions
   int p = get_global_size(0);
   int r = get_global_size(1);
   //int qq = q[0];

   //Number of submatrixes to be processed by each worker (Q dimension)
   int numSubMat = qq/BLOCK_SIZE;

   float4 resp = (float4)(0,0,0,0);
   __local float A[BLOCK_SIZE][BLOCK_SIZE];
   __local float B[BLOCK_SIZE][BLOCK_SIZE];

   for (int k=0; k<numSubMat; k++)
   {
       //Copy submatrixes to local memory. Each worker copies one element
       //Notice that A[i,k] accesses elements starting from M[BLOCK_SIZE*i, BLOCK_SIZE*j]
       A[idX][idY] = M1[BLOCK_SIZE*i + idX + p*(BLOCK_SIZE*k+idY)];
       B[idX][idY] = M2[BLOCK_SIZE*k + idX + qq*(BLOCK_SIZE*j+idY)];
       barrier(CLK_LOCAL_MEM_FENCE);

       for (int k2 = 0; k2 < BLOCK_SIZE; k2+=4) 
       {
            float4 temp1=(float4)(A[idX][k2],A[idX][k2+1],A[idX][k2+2],A[idX][k2+3]);
            float4 temp2=(float4)(B[k2][idY],B[k2+1][idY],B[k2+2][idY],B[k2+3][idY]);
            resp += temp1 * temp2;
       }
       barrier(CLK_LOCAL_MEM_FENCE);
   }

   M3[BLOCK_SIZE*i + idX + p*(BLOCK_SIZE*j+idY)] = resp.x+resp.y+resp.z+resp.w;

}