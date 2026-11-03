/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TRITON_ADAPTER_ACCRESETSELECTCONVERTER_H
#define TRITON_ADAPTER_ACCRESETSELECTCONVERTER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace TTOpConverters {

/// Rewrites the persistent-kernel accumulator reset idiom
///
///   %sel = arith.select %cond, %zeroSplat, %acc : tensor<...>
///
/// into a conditional in-place fill placed after the other consumers of %acc:
///
///   %sel = scf.if %cond {
///     %z = linalg.fill ins(%zeroScalar) outs(%acc)
///     scf.yield %z
///   } else {
///     scf.yield %acc
///   }
///
/// The tensor select would otherwise force the downstream compiler to
/// materialize the splat, the matmul result and the select result as distinct
/// UB buffers, which breaks in-place accumulation (`remain_in_l0c`) and can
/// overflow the UB budget for large tiles. Both branches of the rewritten
/// scf.if yield the same underlying buffer, so no extra buffer is born.
struct AccResetSelectConverter
    : public mlir::OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace TTOpConverters

#endif // TRITON_ADAPTER_ACCRESETSELECTCONVERTER_H
