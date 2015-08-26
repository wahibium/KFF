#ifndef ACTIVITYFACTORGENERATOR_H_INCLUDED
#define ACTIVITYFACTORGENERATOR_H_INCLUDED

#include <traceGenerator.h>
#include <fstream>

namespace meta {

	/*!
		Base class for generating traces
	*/
	class ActivityFactorGenerator: public TraceGenerator {	
	public:
	
	public:
		ActivityFactorGenerator();
		virtual ~ActivityFactorGenerator();

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
		size_t invocations;
		std::string kernelName;
		ir::Dim3 blockDim;
		size_t dynamicInstructions;
		size_t threadInstructions;
	};
}

#endif

