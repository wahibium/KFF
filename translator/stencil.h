#ifndef _STENCIL_H_
#define _STENCIL_H_

#include "common.h"

namespace kff {
namespace translator {

bool AnalyzeStencilIndex(SgExpression *arg, StencilIndex &idx,
                         SgFunctionDeclaration *kernel);
void AnalyzeStencilRange(StencilMap &sm, TranslationContext &tx);

//void AnalyzeEmit(SgFunctionDeclaration *func);

void AnalyzeGet(SgNode *top_level_node,
                TranslationContext &tx);
void AnalyzeEmit(SgNode *top_level_node,
                 TranslationContext &tx);

bool AnalyzeGetArrayMember(SgDotExp *get, SgExpressionVector &indices,
                           SgExpression *&parent);


} // namespace translator
} // namespace kff


#endif 
