#ifndef EXCEPTION_H_INCLUDED
#define EXCEPTION_H_INCLUDED

#include <exception>
#include <string>


/*! \brief a namespace for common classes and functions */
namespace meta
{
	/*! \brief An Exception with a variable message */
	class Exception : public std::exception
	{
		public:
			Exception( const std::string& message );
			virtual ~Exception() throw();
			virtual const char* what() const throw();

		private:
			std::string _message;		
	};
}

#endif
