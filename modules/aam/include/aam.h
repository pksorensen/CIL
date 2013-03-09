#ifndef __CIL_AAM_H__
#define __CIL_AAM_H__

#include <cil/core/types_c.h>

namespace CIL
{
	namespace AAM
	{
		struct AAMParameters
		{

		};

		class CIL_EXPORTS Amm
		{
		public:
			Amm();
			virtual ~Amm();

			virtual void build();
			
		}
	}
}

#endif
