#ifndef AST_H_
#define AST_H_

#include "common.h"

namespace kff {
namespace translator {

int RemoveRedundantVariableCopy(SgNode *scope);
int RemoveUnusedFunction(SgNode *scope);

}  // namespace translator
}  // namespace kff

#endif


