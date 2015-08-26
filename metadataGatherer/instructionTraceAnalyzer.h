#ifndef INTSTRUCTION_TRACE_ANALYZER_H_INCLUDED
#define INTSTRUCTION_TRACE_ANALYZER_H_INCLUDED

#include <instructionTraceGenerator.h>
#include <map>
#include <deque>

namespace meta
{
	/*! \brief Provides the ability to inspect a database created by a 
		BranchTraceGenerator
	*/
	class InstructionTraceAnalyzer {
		private:

			typedef std::deque< KernelEntry > KernelVector;
			typedef std::map< std::string, KernelVector > KernelMap;
			
		private:
			KernelMap _kernels; //! Entries for the kernel traces
		
		public:
		
			/*! \brief The constructor loads a database */
			InstructionTraceAnalyzer(const std::string& database);
			
			/*!
				\brief List all of the kernel traces contained in the database.
			*/
			void list() const;
			
			/*!
				\brief compute the histogram of instructions by (functional unit, opcode) for each
					kernel
			*/
			void instructions_by_kernel(bool pyList = false) const;

			/*!
				\brief compute the histogram of instructions by (functional unit, opcode) for each
					application over all its kernels
			*/
			void instructions_by_application(bool pyList = false) const;
	
	};
}

int main( int argc, char** argv );

#endif

