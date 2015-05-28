#ifndef _COMMON_H_
#define _COMMON_H_


namespace kff {
namespace translator {

SgType *BuildInt32Type(SgScopeStatement *scope=NULL);
SgType *BuildInt64Type(SgScopeStatement *scope=NULL);
SgType *BuildIndexType(SgScopeStatement *scope=NULL);
SgType *BuildIndexType2(SgScopeStatement *scope=NULL);

SgExpression *BuildIndexVal(PSIndex v);

SgType *BuildPSOffsetsType();
SgVariableDeclaration *BuildPSOffsets(std::string name,
                                      SgScopeStatement *scope,
                                      __PSOffsets &v);

SgType *BuildPSGridRangeType();
SgVariableDeclaration *BuildPSGridRange(std::string name,
                                        SgScopeStatement *block,
                                        __PSGridRange &v);

SgExpression *BuildFunctionCall(const std::string &name,
                                SgExpression *arg1);

SgType *GetBaseType(SgType *ty);
  
} // namespace translator
} // namespace kff



#endif
