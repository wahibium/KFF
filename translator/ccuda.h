#ifndef _CCUDA_H_
#define _CCUDA_H_

namespace kff {
namespace translator {

class CUDATranslator {
 public:
  CUDATranslator(const Configuration &config);
  virtual ~CUDATranslator() {}
  
 protected:  
  /** hold __cuda_block_size_struct type */
  SgType *cuda_block_size_type_;
  
  virtual void FixAST();


  virtual CUDABuilderInterface *builder() {
    return dynamic_cast<CUDABuilderInterface*>(rt_builder_);
  }
  
 public:

  virtual void SetUp(SgProject *project, TranslationContext *context,
                     BuilderInterface *rt_builder);

  virtual void appendNewArgExtra(SgExprListExp *args,
                                 Grid *g,
                                 SgVariableDeclaration *dim_decl);


  virtual void ProcessUserDefinedPointType(SgClassDeclaration *grid_decl,
                                           GridType *gt);

  virtual void TranslateKernelDeclaration(SgFunctionDeclaration *node);
  virtual void TranslateGet(SgFunctionCallExp *node,
                            SgInitializedName *gv,
                            bool is_kernel, bool is_periodic);
  virtual void TranslateGetForUserDefinedType(
      SgDotExp *node, SgPntrArrRefExp *array_top);
  virtual void TranslateEmit(SgFunctionCallExp *node,
                             GridEmitAttribute *attr);
  virtual void TranslateFree(SgFunctionCallExp *node,
                             GridType *gt);
  virtual void TranslateCopyin(SgFunctionCallExp *node,
                               GridType *gt);
  virtual void TranslateCopyout(SgFunctionCallExp *node,
                                GridType *gt);

  virtual void FixGridType(const string &real_type_name);
};

} // namespace translator
} // namespace kff


#endif
