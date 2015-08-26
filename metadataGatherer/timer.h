#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <lowLevelTimer.h>
#include <string>

/*!
	\brief a namespace for hydrazine classes and functions
*/
namespace meta
{


	class Timer : public LowLevelTimer
	{
		public:	
			std::string toString() const;
	};

}

#endif

