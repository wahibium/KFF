#include <programStructureGraph.h>

namespace analysis
{

/*! \brief Implements a naive mapping over existing basic blocks */
class DefaultProgramStructure : public ProgramStructureGraph
{
public:
	DefaultProgramStructure(ir::ControlFlowGraph& cfg);

};

}


