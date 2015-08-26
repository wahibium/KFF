#include <analysis.h>

namespace analysis
{

/*! \brief Discover all control and data dependences in a kernel */
class DependenceAnalysis
: public PTXInstructionDependenceGraph, public KernelAnalysis
{
public:
	DependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);

};

}


