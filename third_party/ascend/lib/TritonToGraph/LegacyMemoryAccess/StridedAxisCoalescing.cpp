/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.h"

#include "mlir/IR/BuiltinOps.h"

namespace StridedAxisCoalescing {

using namespace mlir;

// Upstream Triton removed block pointers from the IR ("block pointer is
// python-only": MakeTensorPtrOp, AdvanceOp and TensorPtrType were deleted).
// This pass folded strided 1D block-pointer load/store chains into 2D tiles;
// with no block pointers left in the IR its seeds can never match, so it is a
// no-op. Kept as a stub so the LayoutMemoryCompatibilityPass pipeline is
// unchanged.
void rewriteStridedAxisCoalesce(ModuleOp moduleOp) { (void)moduleOp; }

} // namespace StridedAxisCoalescing
