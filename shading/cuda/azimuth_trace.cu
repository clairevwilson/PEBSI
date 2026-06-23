extern "C" __global__ 
void azimuth_trace(
    float* __restrict__ max_zenith,  // (n_rays, max_ray_len, 2) - (i,j) pairs
    int* __restrict__ max_j, // (n_rays,)
    int* __restrict__ max_i, // (n_rays,)
    const float* __restrict__ dem,      // (ny, nx)
    float j_basis,
    float i_basis,
    float step_size,                 // horizontal distance per step (meters)
    int max_steps,
    int nx, int ny
    ) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= nx) return;

    float base_elev = dem[i * nx + j];
    
    float pixel_elev = 0.0;
    float pixel_dist = 0.0;

    float current_j = j;
    float current_i = i;

    float maximum_zenith = 0.0f;
    int maximum_j = j;
    int maximum_i = i;

    float zenith = 0.0f;
    int round_j = j;
    int round_i = i;

    int step_count = 0;

    while (step_count < max_steps) {
    step_count++;
    current_j += j_basis*step_size;
    current_i += i_basis*step_size;
    pixel_dist += step_size;

	round_j = roundf(current_j);
	round_i = roundf(current_i);

	if (round_i < 0 || round_i >= ny || round_j < 0 || round_j >= nx) break;

	float fj = current_j;
        float fi = current_i;
	int j0 = (int)floorf(fj);
	int i0 = (int)floorf(fi);
	int j1 = j0 + 1;
	int i1 = i0 + 1;
	if (i0 < 0 || i1 >= ny || j0 < 0 || j1 >= nx) break;
	float wj = fj - j0;
	float wi = fi - i0;
	pixel_elev = (1-wi)*(1-wj)*dem[i0*nx+j0]
		   + (1-wi)*wj    *dem[i0*nx+j1]
		   + wi*(1-wj)    *dem[i1*nx+j0]
		   + wi*wj        *dem[i1*nx+j1];

	//pixel_elev = dem[round_i*nx + round_j];
        zenith = (pixel_elev - base_elev) / pixel_dist;
	if (zenith > maximum_zenith) { 
	    maximum_zenith = zenith;
	    maximum_j = round_j;
	    maximum_i = round_i;
	}
    }

    max_zenith[i * nx + j] = maximum_zenith;
    max_j[i * nx + j] = maximum_j;
    max_i[i * nx + j] = maximum_i;
}