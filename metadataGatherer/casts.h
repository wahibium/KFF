#ifndef CASTS_H_INCLUDED
#define CASTS_H_INCLUDED


namespace meta
{
	template< typename To, typename From >
	class UnionCast
	{
		public:
			union
			{
				To to;
				From from;
			};
	};
	
	
	template< typename To, typename From >
	To bit_cast( const From & from )
	{
		UnionCast< To, From > cast;
		cast.to = 0;
		cast.from = from;
		return cast.to;
	}
	
	template< typename To, typename From >
	void bit_cast(To& to, const From& from)
	{
		to = bit_cast<To, From>(from);
	}
}

#endif

