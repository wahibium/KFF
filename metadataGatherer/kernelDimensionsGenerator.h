#ifndef KERNELDIM_TRACE_GENERATOR_H_INCLUDED
#define KERNELDIM_TRACE_GENERATOR_H_INCLUDED

#include <traceGenerator.h>
#include <kernelEntry.h>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace meta {

	/*!
		\brief kernel dimensions trace generator
	*/
	class KernelDimensionsGenerator : public TraceGenerator {
	public:
	
		/*!
			header for InstructionTraceGenerator
		*/
		class Header {
		public:
			Header();
			
		public:
		
			TraceFormat format;

			ir::Dim3 block;
			
			ir::Dim3 grid;
			
		};	
	
	public:

		/*!
			default constructor
		*/
		KernelDimensionsGenerator();

		/*!
			\brief destructs instruction trace generator
		*/
		virtual ~KernelDimensionsGenerator();

		/*!
			\brief called when a traced kernel is launched to retrieve some 
				parameters from the kernel
		*/
		virtual void initialize(const executive::ExecutableKernel& kernel);

		/*!
			\brief Called whenever an event takes place.

			Note, the const reference 'event' is only valid until event() 
			returns
		*/
		virtual void event(const TraceEvent & event);
		
		/*! 
			\brief Called when a kernel is finished. There will be no more 
				events for this kernel.
		*/
		virtual void finish();
		
	public:
	
		Header _header;
		
		KernelEntry _entry;

		static unsigned int _counter;	
	};
}

namespace boost
{
	namespace serialization
	{		
		template< class Archive >
		void serialize( Archive& ar, 
			trace::KernelDimensionsGenerator::Header& header, 
			const unsigned int version )
		{
			ar & header.format;
			
			ar & header.block.x;
			ar & header.block.y;
			ar & header.block.z;
			
			ar & header.grid.x;
			ar & header.grid.y;
			ar & header.grid.z;
		}
	}
}

#endif

