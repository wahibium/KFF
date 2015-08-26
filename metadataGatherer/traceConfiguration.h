#ifndef TRACE_CONFIGURATION_H_INCLUDED
#define TRACE_CONFIGURATION_H_INCLUDED

#include <warpSynchronousGenerator.h>
#include <memoryTraceGenerator.h>
#include <instructionTraceGenerator.h>
#include <parallelismTraceGenerator.h>
#include <sharedComputationGenerator.h>
#include <branchTraceGenerator.h>
#include <cacheSimulator.h>
#include <performanceBoundGenerator.h>
#include <convergenceGenerator.h>
#include <loadBalanceGenerator.h>
#include <controlFlowVisualizer.h>
#include <basicBlockCountGenerator.h>
#include <x86TraceGenerator.h>
#include <temporalSIMTGenerator.h>
#include <activityFactorGenerator.h>

namespace meta
{
	/*! \brief A singleton used to create instances of trace generators */
	class TraceConfiguration
	{
	public:
		//! \brief configuration for the warp synchronous trace generator
		class WarpSynchronous 
		{
		public:
			WarpSynchronous() : enabled(false), emitHotPaths(false),
				csv("warpsynchronous.csv") { }
			
			//! \brief whether trace generator is enabled
			bool enabled;

			//! \brief emits dot file visualizing hot paths
			bool emitHotPaths;
			
			//! \brief path to CSV collecting results
			std::string csv;
		};
		
		/*! \brief configuration properties for 
				trace::PerformanceBoundGenerator trace generator
		*/
		class PerformanceBound {
		public:
			PerformanceBound(): enabled(false),
				protocol(PerformanceBoundGenerator::Protocol_sm_20), 
				render(false),
				outputFormat(PerformanceBoundGenerator::Output_dot) {} 
		
			/*! \brief Enable the performance bound trace generator */
			bool enabled;
			
			/*! \brief Protocol for memory coalescing */
			PerformanceBoundGenerator::CoalescingProtocol protocol;
			
			/*! \brief if true, a PDF is constructed from the DOT file */
			bool render;
			
			PerformanceBoundGenerator::OutputFormat outputFormat;
		};
		
		/*! \brief configuration properties for ConvergenceGenerator */
		class Convergence {
		public:
			Convergence() : enabled(false), logfile("convergence"),
				dot(false), render(false) { }
		
			//! \brief indicates trace generator is enabled
			bool enabled;
			
			//! \brief log file to append results
			std::string logfile;		
			
			//! \brief emit CFG as dot file
			bool dot;	
			
			//! \brief if true, a PDF is constructed from the DOT file
			bool render;
		};
		
		/*! \brief configuration of ControlFlowVisualizer */
		class ControlFlow {
		public:
			ControlFlow(): enabled(false), allInstructions(false) { }
			
			//! \brief whether its enabled
			bool enabled;
			
			//! \brief if false, only branch and reconverge instructions are displayed
			bool allInstructions;
		};
		
		class CacheSimulator {
		public:
			//! \brief Enable the cache simulator
			bool enabled;
			
			//! \brief How many blocking cycles for dirty writebacks
			unsigned int writebackTime;
			
			//! \brief total cache size in bytes
			unsigned int cacheSize;
			
			//! \brief total line size in bytes
			unsigned int lineSize;
			
			//! \brief access time per hit
			unsigned int hitTime;
			
			//! \brief access time per miss
			unsigned int missTime;
			
			//! \brief associativity
			unsigned int associativity;
			
			//! \brief Cache instruction memory rather than data memory
			bool instructionMemory;
			
		};
		
		class TemporalSIMT {
		public:
			bool enabled;
		
			//! number of threads in a SIMT warp
			int warpSize;
		
			//! number of functional unit lanes per SIMD cluster (decoder+controller+simdWidthxALUs)
			int simdWidth;
		
			//! number of warps that can simultaneously issue instructions (total ALUs = simdIssueCount * simdWidth)
			int simdIssueCount;
		};

	public:
		static TraceConfiguration Singleton;

	public:
		//! \brief path to trace generation database
		std::string database;

		//! \brief indicates traces are in place
		bool inPlaceTraces;

		//! \brief trace::MemoryTraceGenerator
		bool memory;

		//! \brief trace::BranchTraceGenerator
		bool branch;

		//! \brief trace::SharedComputationTraceGenerator
		bool sharedComputation;

		//! \brief trace::ParallelismTraceGenerator
		bool parallelism;

		//! \brief trace::InstructionTraceGenerator
		bool instruction;

		//! \brief warp synchronous trace generator
		WarpSynchronous warpSynchronous;
		
		//! \brief performance bound generator
		PerformanceBound performanceBound;
		
		//! \brief cache simulator trace generator
		CacheSimulator cacheSimulator;
		
		//! \brief trace::ConvergenceGenerator
		Convergence convergence;
		
		//! \brief trace::LoadBalanceGenerator
		bool loadBalance;

		//! \brief trace:X86TraceGenerator
		bool x86trace;
		
		ControlFlow controlFlowVisualizer;
		
		bool basicBlockCount;
		
		TemporalSIMT temporalSIMT;
	
		bool activityFactor;
	
	public:
		TraceConfiguration();

	private:
		trace::MemoryTraceGenerator _memoryTraceGenerator;
		trace::InstructionTraceGenerator _instructionTraceGenerator;
		trace::SharedComputationGenerator _sharedComputationGenerator;
		trace::BranchTraceGenerator _branchTraceGenerator;
		trace::ParallelismTraceGenerator _parallelismTraceGenerator;
		trace::CacheSimulator _cacheSimulator;
		trace::WarpSynchronousGenerator _warpSynchronous;
		trace::PerformanceBoundGenerator _performanceBound;
		trace::ConvergenceGenerator _convergence;
		trace::LoadBalanceGenerator _loadBalance;
		trace::ControlFlowVisualizer _controlFlowVisualizer;
		trace::BasicBlockCountGenerator _basicBlockCountGenerator;
		trace::X86TraceGenerator _x86TraceGenerator;
		trace::TemporalSIMTGenerator _temporalSIMTGenerator;
		trace::ActivityFactorGenerator _activityFactorGenerator;
	};
}

#endif

