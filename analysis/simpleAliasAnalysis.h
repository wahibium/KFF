#include <analysis.h>

// Forward Declarations
namespace ir { class Instruction; }

namespace analysis
{

/*! \brief A class for performing simple alias analysis. */
class SimpleAliasAnalysis : public KernelAnalysis
{
public:
	SimpleAliasAnalysis();
	
public:
	#ifndef _WIN32
	SimpleAliasAnalysis& operator=(const SimpleAliasAnalysis&) = delete;
	SimpleAliasAnalysis(const SimpleAliasAnalysis&) = delete;
	#endif
	
public:
	/*! \brief Run the analysis over a specified kernel */
	void analyze(ir::IRKernel& kernel);

public:
	/*! \brief Is it impossible for any store that can reach this load to
			alias this load?
	*/
	bool cannotAliasAnyStore(const ir::Instruction* load);
	/*! \brief Is it possible for these instructions to alias? */
	bool canAlias(const ir::Instruction* store, const ir::Instruction* load);

private:
	bool          _aStoreCanReachThisFunction;
	ir::IRKernel* _kernel;

};

}

