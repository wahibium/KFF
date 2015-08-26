#include <analysis.h>

namespace analysis
{

/*! \brief Discover all data dependences in a kernel */
class DataDependenceAnalysis
: public KernelAnalysis, public PTXInstructionDependenceGraph
{
public:
	DataDependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);

};

}


