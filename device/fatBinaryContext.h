#ifndef CUDA_FATBINARYCONTEXT_H_INCLUDED
#define CUDA_FATBINARYCONTEXT_H_INCLUDED

#include <cudaFatBinary.h>

// Standard Library Includes
#include <vector>

namespace cuda {
	/*!	\brief Class allowing sharing of a fat binary among threads	*/
	class FatBinaryContext {
	public:
		FatBinaryContext(const void *);
		FatBinaryContext();
	
		//! pointer to CUBIN structure
		const void *cubin_ptr;
		
	public:
		const char *name() const;
		const char *ptx() const;

	private:
		void _decompressPTX(unsigned int compressedBinarySize);

	private:
		const char* _name;
		const char* _ptx;

	private:
		typedef std::vector<char> ByteVector;

	private:
		ByteVector _decompressedPTX;
	
	};
}

#endif

