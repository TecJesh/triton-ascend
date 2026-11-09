#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

// Stand-in NPU result/stream types for the AOT C API (see
// launcher_src/launcher_cann.h).
#include "launcher_cann.h"

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
NPUresult{_placeholder} {kernel_name}(NPUstream stream, {signature});
