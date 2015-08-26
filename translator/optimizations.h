#ifndef _OPTIMIZATIONS_H_
#define _OPTIMIZATIONS_H_

#include "common.h"

namespace kff {
namespace translator {
namespace optimizer {

//! Find innermost kernel loops
extern vector<SgForStatement*> FindInnermostLoops(SgNode *proj);

//! Find expressions that are assigned to variable v
extern void GetVariableSrc(SgInitializedName *v,
                           vector<SgExpression*> &src_exprs);

//! Simple dead code elimination
extern bool EliminateDeadCode(SgStatement *stmt);

//! Returns a single source expression for a variable if statically determined
SgExpression *GetDeterministicDefinition(SgInitializedName *var);

} // namespace optimizer
} // namespace translator
} // namespace kff

#endif

