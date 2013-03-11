
#include <Random123/threefry.h>


__kernel void matrix_random_fill(unsigned seed, __global float *matrix, const int num) {
    
	unsigned idx = get_global_id(0)*4;
	
	threefry4x32_key_t k = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    threefry4x32_ctr_t c = {{idx, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

	union {
	    threefry4x32_ctr_t c;
	    float4 f;
	} u;
	u.c = threefry4x32(c, k);
	
	if(idx < num)
	matrix[idx++] = u.f.x;
	if(idx < num)
	matrix[idx++] = u.f.y;
	if(idx < num)
	matrix[idx++] = u.f.z;
	if(idx < num)
	matrix[idx] = u.f.w;
};

//-cl-fast-relaxed-math -DMAC