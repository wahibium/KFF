#include <analysisFactory.h>
#include <dataflowGraph.h>
#include <divergenceAnalysis.h>
#include <affineAnalysis.h>
#include <controlTree.h>
#include <dominatorTree.h>
#include <postdominatorTree.h>
#include <structuralAnalysis.h>
#include <threadFrontierAnalysis.h>
#include <loopAnalysis.h>
#include <convergentRegionAnalysis.h>
#include <simpleAliasAnalysis.h>
#include <cycleAnalysis.h>
#include <safeRegionAnalysis.h>
#include <controlDependenceAnalysis.h>
#include <dataDependenceAnalysis.h>
#include <dependenceAnalysis.h>
#include <memoryDependenceAnalysis.h>
#include <hammockGraphAnalysis.h>

namespace analysis 
{

Analysis* AnalysisFactory::createAnalysis(const std::string& name,
	const StringVector& options)
{
	Analysis* analysis = nullptr;

	if(name == "ControlTreeAnalysis")
	{
		analysis = new ControlTree;
	}
	else if(name == "DominatorTreeAnalysis")
	{
		analysis = new DominatorTree;
	}
	else if(name == "PostDominatorTreeAnalysis")
	{
		analysis = new PostdominatorTree;
	}
    else if(name == "DataflowGraphAnalysis")
	{
		auto dfg = new DataflowGraph;
		
		dfg->setPreferredSSAType(analysis::DataflowGraph::None);
				
		analysis = dfg;
	}
	else if(name == "DivergenceAnalysis")
	{
		analysis = new DivergenceAnalysis;
	}
	else if(name == "AffineAnalysis")
	{
		analysis = new AffineAnalysis;
	}
	else if(name == "StructuralAnalysis")
	{
		analysis = new StructuralAnalysis;
	}
	else if(name == "ThreadFrontierAnalysis")
	{
		analysis = new ThreadFrontierAnalysis;
	}
	else if(name == "LoopAnalysis")
	{
		analysis = new LoopAnalysis;
	}
	else if(name == "ConvergentRegionAnalysis")
	{
		analysis = new ConvergentRegionAnalysis;
	}
	else if(name == "SafeRegionAnalysis")
	{
		analysis = new SafeRegionAnalysis;
	}
	else if(name == "CycleAnalysis")
	{
		analysis = new CycleAnalysis;
	}
	else if(name == "SimpleAliasAnalysis")
	{
		analysis = new SimpleAliasAnalysis;
	}
	else if(name == "ControlDependenceAnalysis")
	{
		analysis = new ControlDependenceAnalysis;
	}
	else if(name == "DataDependenceAnalysis")
	{
		analysis = new DataDependenceAnalysis;
	}
	else if(name == "MemoryDependenceAnalysis")
	{
		analysis = new MemoryDependenceAnalysis;
	}
	else if(name == "DependenceAnalysis")
	{
		analysis = new DependenceAnalysis;
	}
	else if(name == "HammockGraphAnalysis")
	{
		analysis = new HammockGraphAnalysis;
	}

	if(analysis != nullptr)
	{
		analysis->configure(options);
	}
	
	return analysis;
}

}

