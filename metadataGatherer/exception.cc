#ifndef EXCEPTION_CPP_INCLUDED
#define EXCEPTION_CPP_INCLUDED

#include <exception.h>


namespace meta
{
	const char* Exception::what() const throw()
	{
		return _message.c_str();
	
	}

	Exception::~Exception() throw()
	{
	
	}

	Exception::Exception( const std::string& m ) :
		_message( m )
	{
	
		
	
	}
}

#endif

