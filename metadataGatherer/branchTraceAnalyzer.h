#ifndef BRANCH_TRACE_ANALYZER_H_INCLUDED
#define BRANCH_TRACE_ANALYZER_H_INCLUDED

#include <branchTraceGenerator.h>
#include <map>
#include <deque>

namespace meta
{

	/*!
		\brief Provides the ability to inspect a database created by a 
		BranchTraceGenerator
	*/
	class BranchTraceAnalyzer
	{
		private:
			typedef std::deque< KernelEntry > KernelVector;
			typedef std::map< std::string, KernelVector > KernelMap;
			
		private:
			KernelMap _kernels; //! Entries for the kernel traces
		
		public:
		
			/*!
				\brief The constructor loads a database
			*/
			BranchTraceAnalyzer(const std::string& database);
			
			/*!
				\brief List all of the kernel traces contained in the database.
				
				Print the results to stdout
			*/
			void list() const;
			
			/*!
				\brief Compute the branch divergence for each trace.
				
				Print the results to stdout
			*/
			void divergence() const;
	
	};

}

int main( int argc, char** argv );

#endif

