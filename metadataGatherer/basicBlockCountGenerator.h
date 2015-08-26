#ifndef BASICBLOCKCOUNTGENERATOR_H_INCLUDED
#define BASICBLOCKCOUNTGENERATOR_H_INCLUDED

// C++ includes
#include <vector>
#include <map>

// Boost includes
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <kernelEntry.h>

/////////////////////////////////////////////////////////////////////////////////////////////////


namespace meta {

	/*!
	
	*/
	class BasicBlockCountGenerator: public trace::TraceGenerator {
	public:
	
		typedef std::vector< size_t > ThreadCounter;
		typedef std::map< std::string, ThreadCounter > BasicBlockCounter;
	
	public:
	
		BasicBlockCountGenerator();
		
		virtual ~BasicBlockCountGenerator();
		
			/*! \brief called when a traced kernel is launched to retrieve some 
				parameters from the kernel
		*/
		virtual void initialize(const executive::ExecutableKernel& kernel);

		/*! \brief Called whenever an event takes place.

			Note, the const reference 'event' is only valid until event() 
			returns
		*/
		virtual void event(const TraceEvent & event);
		
		/*! \brief called when an event is committed
		*/
		virtual void postEvent(const TraceEvent & event);
		
		/*! \brief Called when a kernel is finished. There will be no more 
				events for this kernel.
		*/
		virtual void finish();
		
	public:
		/*! \brief Counter for creating unique file names. */
		static unsigned int _counter;
		
		/*!	\brief Entry for the current kernel	*/
		KernelEntry _entry;
			
		const executive::EmulatedKernel *emulatedKernel;
			
		BasicBlockCounter basicBlockCounter;
		
	};

}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif

