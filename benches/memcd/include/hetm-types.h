#ifndef HETM_TYPES_H_GUARD_
#define HETM_TYPES_H_GUARD_

// TODO: this is Benchmark stuff

/* ################################################################### *
* BANK ACCOUNTS
* ################################################################### */
typedef int /*__attribute__((aligned (64)))*/ account_t;

typedef struct bank {
  account_t *accounts;
  account_t *devAccounts;
  long size;
} bank_t;

typedef struct memcd {
	account_t *key;   /* keys in global memory --> 4B */
  account_t *extraKey; /* 3*4B */
	account_t *val;   /* values in global memory --> 4B */
	account_t *extraVal;   /* 7*4B */
	account_t *ts_CPU;    /* last access TS in global memory */
	account_t *ts_GPU;    /* last access TS in global memory */
	account_t *state; /* state in global memory */
	account_t *setUsage; /* state in global memory */
	unsigned *globalTs;
  long nbSets;
  int nbWays;
} memcd_t;

typedef struct packet {
	int vers;
	long key;
} packet_t;

#endif /* HETM_TYPES_H_GUARD_ */
