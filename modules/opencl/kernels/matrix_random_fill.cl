
#include <Random123/threefry.h>
#include <Random123/u01.h>

__kernel void matrix_random_fill(unsigned seed, __global float *matrix, unsigned num) {
    
	unsigned idx = get_global_id(0)*4;
	
	threefry4x32_key_t k = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    threefry4x32_ctr_t c = {{idx, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
	threefry4x32_ctr_t r;

	r = threefry4x32(c, k);
	
	if(idx < num)
	matrix[idx++] = u01_open_open_32_24(r.v[0])/10 - 0.05f ;
	if(idx < num)
	matrix[idx++] = u01_open_open_32_24(r.v[1])/10 - 0.05f;
	if(idx < num)
	matrix[idx++] = u01_open_open_32_24(r.v[2])/10 -0.05f;
	if(idx < num)
	matrix[idx] = u01_open_open_32_24(r.v[3])/10 -0.05f;
};

//-cl-fast-relaxed-math -DMAC