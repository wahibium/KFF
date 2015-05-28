#include "attribute.h"
#include "common.h"
#include "range.h"

namespace kff {
namespace translator {

const std::string GridCallAttribute::name = "GridCall";

GridCallAttribute::GridCallAttribute(SgInitializedName *grid_var,
                                     KIND k):
    grid_var_(grid_var), kind_(k) {
}

GridCallAttribute::~GridCallAttribute() {}

AstAttribute *GridCallAttribute::copy() {
  return new GridCallAttribute(grid_var_, kind_);
}

bool GridCallAttribute::IsGet() {
  return kind_ == GET;
}

bool GridCallAttribute::IsGetPeriodic() {
  return kind_ == GET_PERIODIC;
}

bool GridCallAttribute::IsEmit() {
  return kind_ == EMIT;
}

void CopyAllAttributes(SgNode *dst, SgNode *src) {
  // ROSE does not seem to have API for locating all attached
  // attributes or copy them all. So, as an ad-hoc work around, list
  // all potentially attahced attributes here to get them copied to
  // the destination node.
  if (rose_util::GetASTAttribute<StencilIndexVarAttribute>(src)) {
    rose_util::CopyASTAttribute<StencilIndexVarAttribute>(
        dst, src, false);
    LOG_DEBUG() << "StencilIndexVarAttribute found at: "
                << src->unparseToString() << "\n";
  }
}

} // namespace translator
} // namespace kff
