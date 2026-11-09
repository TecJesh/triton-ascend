#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <acl/acl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

// Stand-in NPU result/stream types for the AOT C API: the template must not
// depend on <cuda.h>, which NPU machines do not have.
// Guarded so the per-kernel headers and the link.py kernel.h can coexist.
#ifndef TT_AOT_KERNEL_TYPES
#define TT_AOT_KERNEL_TYPES
typedef int NPUresult;
typedef void *NPUstream;
#define ASCEND_SUCCESS 0
#define ASCEND_ERROR_UNKNOWN 1
#define ASCEND_ERROR_INVALID_VALUE 2
#endif

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
NPUresult{_placeholder} {kernel_name}(NPUstream stream, {signature});
