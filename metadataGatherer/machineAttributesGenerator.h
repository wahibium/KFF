#ifndef MACHINEATTRIBUTES_TRACE_GENERATOR_H_INCLUDED
#define MACHINEATTRIBUTES_TRACE_GENERATOR_H_INCLUDED

#include <traceGenerator.h>
#include <kernelEntry.h>


//////////////////////////////////////////////////////////////////////////////////////////////////

namespace meta {

	/*!
		\brief kernel dimensions trace generator
	*/
	class MachineAttributesGenerator : public TraceGenerator {
	public:
	
		/*!
			header for MachineAttributesGenerator
		*/
		class Header {
		public:
			Header();
			
		public:
		
			TraceFormat format;
			executive::Device device;
		};
	
	public:

		/*!
			default constructor
		*/
		MachineAttributesGenerator();

		/*!
			\brief destructs instruction trace generator
		*/
		virtual ~MachineAttributesGenerator();

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
		void serialize( Archive& ar, executive::Device & device, const unsigned int version ) {
			
			ar & device.ISA;
			ar & device.name;
			ar & device.guid;
			ar & device.totalMemory;
			ar & device.multiprocessorCount;
			ar & device.memcpyOverlap;
			ar & device.maxThreadsPerBlock;
		
			ar & device.maxThreadsDim[0];
			ar & device.maxThreadsDim[1];
			ar & device.maxThreadsDim[2];
		
			ar & device.maxGridSize[0];
			ar & device.maxGridSize[1];
			ar & device.maxGridSize[2];
		
			ar & device.sharedMemPerBlock;
			ar & device.totalConstantMemory;
			ar & device.SIMDWidth;
			ar & device.memPitch;
			ar & device.regsPerBlock;
			ar & device.clockRate;
			ar & device.textureAlign;
			ar & device.major;
			ar & device.minor;
		}
			
		template< class Archive >
		void serialize( Archive& ar, trace::MachineAttributesGenerator::Header& header, 
			const unsigned int version ) {
			
			ar & header.format;
			ar & header.device;
		}
	}
}

#endif

