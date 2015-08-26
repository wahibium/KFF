#include <analysis.h>


namespace analysis
{

/*! \brief A class for assigning unique predicate mask variables to
	control flow graph regions.
*/
class ConvergentRegionAnalysis: public KernelAnalysis
{
public:
	typedef ir::ControlFlowGraph CFG;

	typedef CFG::const_iterator const_iterator;
	
	typedef unsigned int Region;

	class RegionContainer
	{
	public:
		RegionContainer* link;
		Region           regionId;
	};

public:
	/*! \brief Create the analysis */
	ConvergentRegionAnalysis();

	/*! \brief Computes an up to date set of regions */
	void analyze(ir::IRKernel& kernel);

public:
	/*! \brief Get the region of the specified block */
	Region getRegion(const_iterator block) const;
	
private:
	typedef std::unordered_map<const_iterator, Region> RegionMap;

private:
	RegionMap _regions;
	
};

}

