#include <analysis.h>

namespace analysis
{

/*! \brief Discover control dependences in the program */
class ControlDependenceAnalysis
: public KernelAnalysis, public PTXInstructionDependenceGraph
{
public:
	ControlDependenceAnalysis();

public:
	void analyze(ir::IRKernel& kernel);
};

}


