#include <dependenceAnalysis.h>
#include <controlDependenceAnalysis.h>
#include <dataDependenceAnalysis.h>
#include <memoryDependenceAnalysis.h>
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

DependenceAnalysis::DependenceAnalysis()
: KernelAnalysis("DependenceAnalysis",
	{"DataDependenceAnalysis", "ControlDependenceAnalysis",
		"MemoryDependenceAnalysis"})
{

}

typedef DependenceAnalysis::Node Node;

typedef DependenceAnalysis::InstructionToNodeMap InstructionToNodeMap;

static void addEdges(Node& node, const Node& existingNode,
	const InstructionToNodeMap& instructionToNodes);

void DependenceAnalysis::analyze(ir::IRKernel& kernel)
{
	report("Running dependence analysis on kernel " << kernel.name);
	
	auto controlDependenceAnalysis = static_cast<ControlDependenceAnalysis*>(
		getAnalysis("ControlDependenceAnalysis")); 
	auto dataDependenceAnalysis = static_cast<DataDependenceAnalysis*>(
		getAnalysis("DataDependenceAnalysis")); 
	auto memoryDependenceAnalysis = static_cast<MemoryDependenceAnalysis*>(
		getAnalysis("MemoryDependenceAnalysis")); 
		
	for(auto& node : *controlDependenceAnalysis)
	{
		auto newNode = _nodes.insert(_nodes.end(), Node(node.instruction));
	
		_instructionToNodes.insert(std::make_pair(node.instruction, newNode));
	}
	
	for(auto& node : *this)
	{
		auto controlDependenceNode = controlDependenceAnalysis->getNode(
			node.instruction);
	
		addEdges(node, *controlDependenceNode, _instructionToNodes);

		auto dataDependenceNode = dataDependenceAnalysis->getNode(
			node.instruction);
	
		addEdges(node, *dataDependenceNode, _instructionToNodes);
		
		auto memoryDependenceNode = memoryDependenceAnalysis->getNode(
			node.instruction);
		
		if(memoryDependenceNode != memoryDependenceAnalysis->end())
		{
			addEdges(node, *memoryDependenceNode, _instructionToNodes);
		}
	}
}

static void addEdges(Node& node, const Node& existingNode,
	const InstructionToNodeMap& instructionToNodes)
{
	for(auto& predecessor : existingNode.predecessors)
	{
		auto predecessorNode = instructionToNodes.find(
			predecessor->instruction);
	
		assert(predecessorNode != instructionToNodes.end());
		
		node.predecessors.push_back(predecessorNode->second);
	}
	
	for(auto& successor : existingNode.successors)
	{
		auto successorNode = instructionToNodes.find(successor->instruction);
	
		assert(successorNode != instructionToNodes.end());
		
		report(" " << node.instruction->toString() << " -> "
			<< successor->instruction->toString());
	
		node.successors.push_back(successorNode->second);
	}
	
}

}


