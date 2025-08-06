#ifndef MMU_H
#define MMU_H
#include <ap_fixed.h>
typedef ap_fixed<8, 3> DTYPE;
#define D_IN 4
#define D_OUT 4
#define D_TEMP (D_OUT/2)

void mmu(DTYPE in[D_IN], DTYPE weight[D_IN][D_OUT], DTYPE out[D_OUT]);
#endif
