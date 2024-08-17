/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

#include "selective_scan_bwd_kernel_ndstate.cuh"

template void selective_scan_bwd_cuda<1, float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::Half, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_cuda<1, at::BFloat16, float>(SSMParamsBwd &params, cudaStream_t stream);

