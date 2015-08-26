#ifndef TRACE_CONTROLFLOWVISUALIZER_H_INCLUDED
#define TRACE_CONTROLFLOWVISUALIZER_H_INCLUDED

// C++ includes
#include <fstream>

#include <traceGenerator.h>
#include <kernelEntry.h>

namespace meta {

	/*!
		Base class for generating traces
	*/
	class ControlFlowVisualizer: public TraceGenerator {	
	public:
	
	public:
		ControlFlowVisualizer();
		virtual ~ControlFlowVisualizer();

		/*! \brief called when a traced kernel is launched to retrieve some 
				parameters from the kernel
		*/
		virtual void initialize(const executive::ExecutableKernel& kernel);

		/*! \brief Called whenever an event takes place.

			Note, the const reference 'event' is only valid until event() 
			returns
		*/
		virtual void event(const TraceEvent & event);
		
		/*! \brief Called when a kernel is finished. There will be no more 
				events for this kernel.
		*/
		virtual void finish();
		
	
	protected:
			
	public:
	
		bool allInstructions;
	
	public:
		
		//! \brief kernel
		const executive::EmulatedKernel * kernel;

		/*! \brief Counter for creating unique file names. */
		static unsigned int _counter;
		
		/*!	\brief Entry for the current kernel	*/
		KernelEntry _entry;
	
		//! maps the PC of the last instruction of each block to the block's label
		std::map< int, std::string > PCsToBlocks;
	};
}

#endif

