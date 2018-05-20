#include "agmg.h"

void parAlmondPrecon(parAlmond_t *parAlmond, occa::memory o_x, occa::memory o_rhs) {

  agmgLevel *baseLevel = parAlmond->levels[0];
  setupAide options = parAlmond->options;

  if (baseLevel->gatherLevel==true) {// gather rhs
    baseLevel->device_gather(baseLevel->gatherArgs, o_rhs, baseLevel->o_rhs);
  } else {
    baseLevel->o_rhs.copyFrom(o_rhs);
  }

  if (options.compareArgs("PARALMOND CYCLE", "HOST")) {
    //host versions
    baseLevel->o_rhs.copyTo(baseLevel->rhs);
    if(options.compareArgs("PARALMOND CYCLE", "EXACT")) {
      if(parAlmond->ktype == PCG) {
        pcg(parAlmond,1000,1e-8);
      } else if(parAlmond->ktype == GMRES) {
        pgmres(parAlmond,1000,1e-8);
      }
    } else if(options.compareArgs("PARALMOND CYCLE", "KCYCLE")) {
      kcycle(parAlmond, 0);
    } else if(options.compareArgs("PARALMOND CYCLE", "VCYCLE")) {
      vcycle(parAlmond, 0);
    }
    baseLevel->o_x.copyFrom(baseLevel->x);
  } else {
    if(options.compareArgs("PARALMOND CYCLE", "EXACT")){
      if(parAlmond->ktype == PCG) {
        device_pcg(parAlmond,1000,1e-8);
      } else if(parAlmond->ktype == GMRES) {
        device_pgmres(parAlmond,1000,1e-8);
      }
    } else if(options.compareArgs("PARALMOND CYCLE", "KCYCLE")) {
      device_kcycle(parAlmond, 0);
    } else if(options.compareArgs("PARALMOND CYCLE", "VCYCLE")) {
      device_vcycle(parAlmond, 0);
    }
  }

  if (baseLevel->gatherLevel==true) {// scatter solution
    baseLevel->device_scatter(baseLevel->scatterArgs, baseLevel->o_x, o_x);
  } else {
    baseLevel->o_x.copyTo(o_x,baseLevel->Nrows*sizeof(dfloat));
  }
}

parAlmond_t *parAlmondInit(mesh_t *mesh, setupAide options) {

  parAlmond_t *parAlmond = (parAlmond_t *) calloc(1,sizeof(parAlmond_t));

  parAlmond->device = mesh->device;
  parAlmond->defaultStream = mesh->defaultStream;
  parAlmond->dataStream = mesh->dataStream;
  parAlmond->options = options;

  parAlmond->levels = (agmgLevel **) calloc(MAX_LEVELS,sizeof(agmgLevel *));
  parAlmond->numLevels = 0;
  
  if (options.compareArgs("PARALMOND CYCLE", "NONSYM")) {
    parAlmond->ktype = GMRES;  
  } else {
    parAlmond->ktype = PCG;
  }

  buildAlmondKernels(parAlmond);

  //buffer for innerproducts in kcycle
  parAlmond->o_rho  = mesh->device.malloc(3*sizeof(dfloat));

  return parAlmond;
}

void parAlmondAgmgSetup(parAlmond_t *parAlmond,
                         hlong* globalRowStarts,       //global partition
                         dlong nnz,                    //--
                         hlong* Ai,                    //-- Local A matrix data (globally indexed, COO storage, row sorted)
                         hlong* Aj,                    //--
                         dfloat* Avals,                //--
                         bool nullSpace,
                         dfloat nullSpacePenalty){   

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  hlong TotalRows = globalRowStarts[size];
  dlong numLocalRows = (dlong) (globalRowStarts[rank+1]-globalRowStarts[rank]);

  if(rank==0) printf("Setting up AMG...");fflush(stdout);

  csr *A = newCSRfromCOO(numLocalRows,globalRowStarts,nnz, Ai, Aj, Avals);

  //record if there is null space
  parAlmond->nullSpace = nullSpace;
  parAlmond->nullSpacePenalty = nullSpacePenalty;

  //populate null space vector
  dfloat *nullA = (dfloat *) calloc(numLocalRows, sizeof(dfloat));
  for (dlong i=0;i<numLocalRows;i++) nullA[i] = 1/sqrt(TotalRows);

  agmgSetup(parAlmond, A, nullA, globalRowStarts, parAlmond->options);
  
  if(rank==0) printf("done.\n");

  if (parAlmond->options.compareArgs("VERBOSE","TRUE"))
    parAlmondReport(parAlmond);
}

//TODO code this
int parAlmondFree(void* A) {
  return 0;
}
