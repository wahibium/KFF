#include <analysis.h>

namespace analysis
{

/*! \brief Discover memory dependences in the program */
class MemoryDependenceAnalysis
: public KernelAnalysis, public PTXInstructionDependenceGraph
{
public:
	MemoryDependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);
};

}


