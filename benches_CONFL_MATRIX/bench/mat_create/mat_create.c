#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <bsd/stdlib.h>

#define FILENAME_PATTERN "STM_RWSET_thr%i.txt" 
#define OUTGRAPH_PATTERN "./pha%i_G%i_thr%i.txt" 
#define INIT_SIZE 8
#define MAT_POS(_wrt, _rd) mat[_rd*matSize + _wrt]

typedef struct tx_info_ {
  unsigned long *rs_addrs;
  unsigned long *ws_addrs;
  unsigned long ts;
  int counter;
} tx_info_s;

typedef struct tx_loc_ {
  int t/*thread*/;
  int p/*phase*/;
  int x/*transaction*/;
} tx_loc_s;

// per thread > per phase
tx_info_s ***perThreadRWSet = NULL;
int nbThreads = 0;
int nbGraphsPerPhase = 0;
int *nbPhases = 0; // per thread
int **nbTXs = 0; // per thread > per phase
int matSize = 0;
char pathToFiles[1024] = "./";
unsigned char *mat = NULL;
tx_loc_s *selectedTXs = NULL;

static void
handleParameters(int argc, char **argv);

static void
initData();

static void
readFile(FILE *fp, int thrId);

FILE *
selectTXs(int idx, int lookupPhase);

void
printAGraph(FILE *fp);

int
main(int argc, char **argv)
{
  int i, j;

  handleParameters(argc, argv);
  initData();
  
  // build matrix
  for (i = 0; i < nbPhases[0]/*TODO*/; ++i) {
    for (j = 0; j < nbGraphsPerPhase; ++j) {
      printAGraph(selectTXs(j, i));
    }
  }

  return EXIT_SUCCESS;
}

static void
handleParameters(int argc, char **argv)
{
  int i;
  char filename[2048];
  for (i = 1; i < argc; ++i) {
    if (argv[i][0] == '-') {
      switch(argv[i][1]) {
        case 't':
          if (++i == argc) break; // missing number
          nbThreads = atoi(argv[i]);
          nbPhases = (int*)malloc(nbThreads*sizeof(int));
          nbTXs = (int**)malloc(nbThreads*sizeof(int*));
          perThreadRWSet = (tx_info_s***)malloc(nbThreads*sizeof(tx_info_s**));
          memset(perThreadRWSet, 0, nbThreads*sizeof(tx_info_s*));
          memset(nbTXs, 0, nbThreads*sizeof(int*));
          break;
        case 'p':
          if (++i == argc) break; // missing path
          sprintf(pathToFiles, "%s/", argv[i]);
          break;
        case 'n':
          if (++i == argc) break; // missing path
          nbGraphsPerPhase = atoi(argv[i]);
          break;
        case 'g':
          if (++i == argc) break; // missing path
          matSize = atoi(argv[i]);
          mat = (unsigned char*)malloc(sizeof(unsigned char)*matSize*matSize);
          memset(mat, 0, sizeof(unsigned char)*matSize*matSize);
          selectedTXs = (tx_loc_s*)malloc(sizeof(tx_loc_s)*matSize);
          memset(selectedTXs, 0, sizeof(tx_loc_s)*matSize);
          break;
        default:
          break;
      }
    }
  }
  if ( !nbThreads || !matSize ){
    printf("Usage: %s -t <nbThreads> -p <path> -g <graphSize> -n <nbGraphsPerPhase>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  sprintf(filename, "%s" FILENAME_PATTERN, pathToFiles, i);
  printf(
    "Running with: \n"
    "    number of threads: %i\n"
    "    filename pattern : %s\n"
    "    graph size       : %i\n"
    "    graph per phase  : %i\n",
    nbThreads, filename, matSize, nbGraphsPerPhase);
}

static void
initData()
{
  int i, j;
  char filename[2048];
  FILE *fp = NULL;
  int nbTXsPerPhase[128];
  int maxPhases = 0;
  memset(nbTXsPerPhase, 0, sizeof(int)*128);
  for (i = 0; i < nbThreads; ++i) {
    sprintf(filename, "%s" FILENAME_PATTERN, pathToFiles, i);
    // printf("processing file %s", filename);
    // fflush(stdout);
    fp = fopen(filename, "r");
    if (fp) {
      readFile(fp, i);
    } else {
      printf("Cannot open %s\n", filename);
      exit(EXIT_FAILURE);
    }
    if (nbPhases[i] > maxPhases) maxPhases = nbPhases[i];
    for (j = 0; j < nbPhases[i]; ++j) {
      nbTXsPerPhase[j] += nbTXs[i][j];
    }
    // fclose(fp); // crashes... I think it is in EOF
  }
  for (i = 0; i < maxPhases; ++i) {
    printf("Phase%i has %i TXs\n", i, nbTXsPerPhase[i]);
  }
}

static void
readFile(FILE *fp, int thrId)
{
  int i;
  int cTXinThr;
  int cTXclock;
  unsigned long cTXtime;
  int cTXphase = 0;
  int maxPhases = INIT_SIZE;
  int maxTXs = INIT_SIZE;
  int lastPhase = 0;
  int countTXinPhase = 0;
  int firstLine = 1;

  nbPhases[thrId] = 0;
  perThreadRWSet[thrId] = (tx_info_s**)malloc(maxPhases*sizeof(tx_info_s*));
  for (i = 0; i < maxPhases; ++i) {
    perThreadRWSet[thrId][i] = (tx_info_s*)malloc(maxTXs*sizeof(tx_info_s));
  }
  
  while (
    fscanf(fp, "# TX=%i Tinyclock=%i time=%lu phase=%i\n",
      &cTXinThr, &cTXclock, &cTXtime, &cTXphase) == 4
  ){
    // printf(".");
    // fflush(stdout);
    void *addr;
    unsigned long *ptr;
    int size = INIT_SIZE;
    nbPhases[thrId] = cTXphase+1;

    if (cTXphase >= maxPhases) {
      maxPhases <<= 1;
      perThreadRWSet[thrId] = (tx_info_s**)realloc(perThreadRWSet[thrId], maxPhases*sizeof(tx_info_s*));
      for (i = (maxPhases>>1); i < maxPhases; ++i) {
        perThreadRWSet[thrId][i] = (tx_info_s*)malloc(maxTXs*sizeof(tx_info_s));
      }
    }

    if (cTXphase != lastPhase) {
      maxTXs = INIT_SIZE;
      nbTXs[thrId] = (int*)realloc(nbTXs[thrId], nbPhases[thrId]*sizeof(int));
      nbTXs[thrId][lastPhase] = countTXinPhase;
      // printf("  thr%i pha%i TX%i (count=%i)\n", thrId, cTXphase, cTXinThr, countTXinPhase);
      countTXinPhase = 0;
    }

    if (firstLine) {
      while (cTXphase > lastPhase) { // may not start in 0
        nbTXs[thrId][lastPhase] = 0;
        lastPhase++;
      }
      firstLine = 0;
    }

    // TODO: cTXinThr >= maxTXs
    if (countTXinPhase >= maxTXs) {
      maxTXs <<= 1;
      perThreadRWSet[thrId][cTXphase] = (tx_info_s*)realloc(perThreadRWSet[thrId][cTXphase], maxTXs*sizeof(tx_info_s));
    }

    perThreadRWSet[thrId][cTXphase][countTXinPhase].counter = cTXclock;
    perThreadRWSet[thrId][cTXphase][countTXinPhase].ts = cTXtime;
    // printf("  thr%i pha%i TX%i (count=%i)\n", thrId, cTXphase, cTXinThr, countTXinPhase);
    if (fscanf(fp, "WS=%p", &addr) == 1) {
      // there is at least 1 write
      perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs = (unsigned long*)
        malloc(size*sizeof(unsigned long));
      ptr = perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs;
      *(ptr++) = (unsigned long)addr;
      while (fscanf(fp, ",%p", &addr) == 1) {
        *(ptr++) = (unsigned long)addr;
        if (ptr == perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs + size) {
          size <<= 1;
          perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs = (unsigned long*)
            realloc(perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs, size*sizeof(unsigned long));
          ptr = perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs + (size>>1);
        }
      }
      // printf("thr%i pha%i TX%i WS_size=%li\n", thrId, cTXphase, cTXinThr, ptr - perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs);
      *(ptr++) = 0; // TODO: be sure there are no NULLs in the WS
    } else perThreadRWSet[thrId][cTXphase][countTXinPhase].ws_addrs = NULL;

    size = INIT_SIZE;
    if (fscanf(fp, "\nRS=%p", &addr) == 1) {
      // there is at least 1 read
      perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs = (unsigned long*)
        malloc(size*sizeof(unsigned long));
      ptr = perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs;
      *(ptr++) = (unsigned long)addr;
      while (fscanf(fp, ",%p", &addr) == 1) {
        *(ptr++) = (unsigned long)addr;
        if (ptr == perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs + size) {
          size <<= 1;
          perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs = (unsigned long*)
            realloc(perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs, size*sizeof(unsigned long));
          ptr = perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs + (size>>1);
        }
      }
      // printf("thr%i pha%i TX%i RS_size=%li\n", thrId, cTXphase, cTXinThr, ptr - perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs);
      *(ptr++) = 0; // TODO: be sure there are no NULLs in the RS
    } else perThreadRWSet[thrId][cTXphase][countTXinPhase].rs_addrs = NULL;
    fscanf(fp, "\n\n"); // move to the next TX

    if (cTXphase > lastPhase) lastPhase++;
    while (cTXphase > lastPhase) { // may have gaps
      nbTXs[thrId][lastPhase] = 0;
      lastPhase++;
    }

    countTXinPhase++;
  }
  if (!firstLine) { // read some transaction 
    nbTXs[thrId] = (int*)realloc(nbTXs[thrId], nbPhases[thrId]*sizeof(int));
    nbTXs[thrId][lastPhase] = countTXinPhase;
    // printf("---thr%i pha%i TX%i (count=%i)\n", thrId, cTXphase, cTXinThr, countTXinPhase);
    countTXinPhase = 0;
  }
  // printf("\n");
}

FILE *
selectTXs(int idx, int lookupPhase)
{
  int i, j;
  FILE *fp;
  char outFilename[1024];
  int countTXs = 0;
  unsigned rnd;
  
  sprintf(outFilename, OUTGRAPH_PATTERN, lookupPhase, idx, matSize);
  fp = fopen(outFilename, "w+");
  if (!fp) {
    printf("Cannot open file to write graph: %s\n", outFilename);
    exit(EXIT_FAILURE);
  }

  // there are positions left in the graph that need to be filled;
  rnd = arc4random();
  i = 0;
  while (countTXs < matSize) {
    if (nbPhases[i % nbThreads] > lookupPhase && nbTXs[i % nbThreads][lookupPhase] > 0) {
      selectedTXs[countTXs].t = i % nbThreads;
      selectedTXs[countTXs].p = lookupPhase;
      selectedTXs[countTXs].x = rnd % nbTXs[i % nbThreads][lookupPhase]; // TODO: collisions are possible
      // printf("Selected thr%i pha%i tx%i\n", selectedTXs[countTXs].t, selectedTXs[countTXs].p, selectedTXs[countTXs].x);
      countTXs++;
    }
    i++;
    if (i % nbThreads == 0) rnd++;
    if (i < 0) {
      printf("Not enough data for the graph\n");
      exit(EXIT_FAILURE);
    }
  }

  return fp;
}

void
printAGraph(FILE *fp)
{
  int i, j;
  unsigned long *rPtr, *wPtr;
  int r_thr, r_pha, r_tx;
  int w_thr, w_pha, w_tx;
  volatile int countE = 0, matchE = 0;

  memset(mat, 0, sizeof(unsigned char)*matSize*matSize);

  for (i = 0; i < matSize; ++i) {
    w_thr = selectedTXs[i].t;
    w_pha = selectedTXs[i].p;
    w_tx = selectedTXs[i].x;
    wPtr = perThreadRWSet[w_thr][w_pha][w_tx].ws_addrs;
    while (*wPtr != 0) { // there is some address
      for (j = 0; j < matSize; ++j) {
        if (i == j || MAT_POS(i, j)) continue;
        r_thr = selectedTXs[j].t;
        r_pha = selectedTXs[j].p;
        r_tx = selectedTXs[j].x;
        rPtr = perThreadRWSet[r_thr][r_pha][r_tx].rs_addrs;
        if (rPtr == NULL) { // bug!
          rPtr = perThreadRWSet[r_thr][r_pha][r_tx].ws_addrs;
        }
        while (rPtr != NULL && *rPtr != 0) {
          if (*wPtr == *(rPtr++)) {
            // MAT_POS(j/*reader*/, i/*writer*/) = 1; // TODO: does not match below...
            MAT_POS(i, j) = 1;
            countE++;
            break;
          }
        }
      }
      wPtr++;
    }
  }
  printf(" <<<>>> countE=%i\n", countE);

  fprintf(fp, "%i %i\n", matSize, countE);
  for (i = 0; i < matSize; ++i) {
    for (j = 0; j < matSize; ++j) {
      if (i == j) continue;
      // if (MAT_POS(j, i)) {
      if (MAT_POS(i, j)) {
        fprintf(fp, "%i %i\n", i, j);
        matchE++;
      }
    }
  }
  assert(countE == matchE);

  fclose(fp);
}

