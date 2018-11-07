/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus, Rajesh Gandham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "agmg.h"

csr *strong_graph(csr *A, dfloat threshold);
bool customLess(int smax, dfloat rmax, hlong imax, int s, dfloat r, hlong i);
hlong *form_aggregates(agmgLevel *level, csr *C);
void find_aggregate_owners(agmgLevel *level, hlong* FineToCoarse, setupAide options);
csr *construct_interpolator(agmgLevel *level, hlong *FineToCoarse, dfloat **nullCoarseA);
csr *transpose(agmgLevel* level, csr *A, hlong *globalRowStarts, hlong *globalColStarts);
csr *galerkinProd(agmgLevel *level, csr *R, csr *A, csr *P);
void coarsenAgmgLevel(agmgLevel *level, csr **coarseA, csr **P, csr **R, dfloat **nullCoarseA, setupAide options);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//   my prototipes
///////////////////////////////////////////////////////////////////////////////////////////////////////////
csr *strong_graph2(csr *A, dfloat threshold);
hlong *form_aggregates2(agmgLevel *level, csr *C);
csr  *OneSmooth(agmgLevel *level, double w,csr *A,csr *P);
csr *construct_interpolator2(agmgLevel *level, hlong *FineToCoarse, dfloat **nullCoarseA);
hlong *form_aggregates3(agmgLevel *level, csr *C,setupAide options);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void agmgSetup(parAlmond_t *parAlmond, csr *A, dfloat *nullA, hlong *globalRowStarts, setupAide options){

  int rank, size;
  rank = agmg::rank;
  size = agmg::size;


  // approximate Nrows at coarsest level
  int gCoarseSize = 1000;

  int MyKey = 0;
  options.getArgs("MYFUNCTIONS",MyKey);
  
  // if (MyKey == 0) 	 gCoarseSize = 1000;     // default 
  //else 		    	 options.getArgs("FINALCOARSEGRIDSIZE",gCoarseSize);    // get it from setup file
  
  options.getArgs("FINALCOARSEGRIDSIZE",gCoarseSize);    // get it from setup file
  printf("\n+++++++++++++++++++++++++\n gCoarseSize = %d \n+++++++++++++++++++++++++\n",gCoarseSize);
  
   
  double seed = (double) rank;
  srand48(seed);

  agmgLevel **levels = parAlmond->levels;

  int lev = parAlmond->numLevels; //add this level to the end of the chain

  levels[lev] = (agmgLevel *) calloc(1,sizeof(agmgLevel));
  levels[lev]->gatherLevel = false;
  levels[lev]->weightedInnerProds = false;
  parAlmond->numLevels++;

  //copy A matrix and null vector
  levels[lev]->A = A;
  levels[lev]->A->null = nullA;

  levels[lev]->Nrows = A->Nrows;
  levels[lev]->Ncols = A->Ncols;

  
  SmoothType smoothType;
  int ChebyshevIterations=2; //default to degree 2
  if (options.compareArgs("PARALMOND SMOOTHER", "CHEBYSHEV")) {
    smoothType = CHEBYSHEV;
    options.getArgs("PARALMOND CHEBYSHEV DEGREE", ChebyshevIterations);
  } else { //default to DAMPED_JACOBI
    smoothType = DAMPED_JACOBI;
  }
  levels[lev]->ChebyshevIterations = ChebyshevIterations;

  setupSmoother(parAlmond, levels[lev], smoothType);

  levels[lev]->deviceA = newHYB(parAlmond, levels[lev]->A);

  //set operator callback
  void **args = (void **) calloc(2,sizeof(void*));
  args[0] = (void *) parAlmond;
  args[1] = (void *) levels[lev];

  levels[lev]->AxArgs = args;
  levels[lev]->smoothArgs = args;
  levels[lev]->Ax = agmgAx;
  levels[lev]->smooth = agmgSmooth;
  levels[lev]->device_Ax = device_agmgAx;
  levels[lev]->device_smooth = device_agmgSmooth;

  //copy global partiton
  levels[lev]->globalRowStarts = (hlong *) calloc(size+1,sizeof(hlong));
  for (int r=0;r<size+1;r++)
      levels[lev]->globalRowStarts[r] = globalRowStarts[r];

  hlong localSize = (hlong) levels[lev]->A->Nrows;
  hlong globalSize = 0;
  MPI_Allreduce(&localSize, &globalSize, 1, MPI_HLONG, MPI_SUM, agmg::comm);

  //if the system if already small, dont create MG levels
  bool done = false;
  if(globalSize <= gCoarseSize){
    setupExactSolve(parAlmond, levels[lev],parAlmond->nullSpace,parAlmond->nullSpacePenalty);
    //setupSmoother(parAlmond, levels[lev], smoothType);
    done = true;
  }
  while(!done){
    // create coarse MG level
    levels[lev+1] = (agmgLevel *) calloc(1,sizeof(agmgLevel));
    dfloat *nullCoarseA;

    //printf("Setting up coarse level %d\n", lev+1);

    coarsenAgmgLevel(levels[lev], &(levels[lev+1]->A), &(levels[lev+1]->P),
                                  &(levels[lev+1]->R), &nullCoarseA, parAlmond->options);

    //set dimensions of the fine level (max among the A,R ops)
    levels[lev]->Ncols = mymax(levels[lev]->Ncols, levels[lev+1]->R->Ncols);

    parAlmond->numLevels++;

    levels[lev+1]->A->null = nullCoarseA;
    levels[lev+1]->Nrows = levels[lev+1]->A->Nrows;
    levels[lev+1]->Ncols = mymax(levels[lev+1]->A->Ncols, levels[lev+1]->P->Ncols);
    levels[lev+1]->globalRowStarts = levels[lev]->globalAggStarts;
    
    levels[lev+1]->ChebyshevIterations = ChebyshevIterations;

    setupSmoother(parAlmond, levels[lev+1], smoothType);

    levels[lev+1]->deviceA = newHYB (parAlmond, levels[lev+1]->A);
    levels[lev+1]->deviceR = newHYB (parAlmond, levels[lev+1]->R);
    levels[lev+1]->dcsrP   = newDCOO(parAlmond, levels[lev+1]->P);

    //set operator callback
    void **args = (void **) calloc(2,sizeof(void*));
    args[0] = (void *) parAlmond;
    args[1] = (void *) levels[lev+1];

    levels[lev+1]->AxArgs = args;
    levels[lev+1]->coarsenArgs = args;
    levels[lev+1]->prolongateArgs = args;
    levels[lev+1]->smoothArgs = args;

    levels[lev+1]->Ax = agmgAx;
    levels[lev+1]->coarsen = agmgCoarsen;
    levels[lev+1]->prolongate = agmgProlongate;
    levels[lev+1]->smooth = agmgSmooth;

    levels[lev+1]->device_Ax = device_agmgAx;
    levels[lev+1]->device_coarsen = device_agmgCoarsen;
    levels[lev+1]->device_prolongate = device_agmgProlongate;
    levels[lev+1]->device_smooth = device_agmgSmooth;

    const hlong localCoarseDim = (hlong) levels[lev+1]->A->Nrows;
    hlong globalCoarseSize;
    MPI_Allreduce(&localCoarseDim, &globalCoarseSize, 1, MPI_HLONG, MPI_SUM, agmg::comm);

    if(globalCoarseSize <= gCoarseSize || globalSize < 2*globalCoarseSize){
      setupExactSolve(parAlmond, levels[lev+1],parAlmond->nullSpace,parAlmond->nullSpacePenalty);
      //setupSmoother(parAlmond, levels[lev+1], smoothType);
      break;
    }

    globalSize = globalCoarseSize;
    lev++;
  } 
  
  //allocate vectors required
  occa::device device = parAlmond->device;
  for (int n=0;n<parAlmond->numLevels;n++) {
    dlong N = levels[n]->Nrows;
    dlong M = levels[n]->Ncols;

    if ((n>0)&&(n<parAlmond->numLevels)) { //kcycle vectors
      if (M) levels[n]->ckp1 = (dfloat *) calloc(M,sizeof(dfloat));
      if (N) levels[n]->vkp1 = (dfloat *) calloc(N,sizeof(dfloat));
      if (N) levels[n]->wkp1 = (dfloat *) calloc(N,sizeof(dfloat));

      if (M) levels[n]->o_ckp1 = device.malloc(M*sizeof(dfloat),levels[n]->ckp1);
      if (N) levels[n]->o_vkp1 = device.malloc(N*sizeof(dfloat),levels[n]->vkp1);
      if (N) levels[n]->o_wkp1 = device.malloc(N*sizeof(dfloat),levels[n]->wkp1);
    }
    if (M) levels[n]->x    = (dfloat *) calloc(M,sizeof(dfloat));
    if (M) levels[n]->res  = (dfloat *) calloc(M,sizeof(dfloat));
    if (N) levels[n]->rhs  = (dfloat *) calloc(N,sizeof(dfloat));

    if (M) levels[n]->o_x   = device.malloc(M*sizeof(dfloat),levels[n]->x);
    if (M) levels[n]->o_res = device.malloc(M*sizeof(dfloat),levels[n]->res);
    if (N) levels[n]->o_rhs = device.malloc(N*sizeof(dfloat),levels[n]->rhs);
  }
  //buffer for innerproducts in kcycle
  dlong numBlocks = ((levels[0]->Nrows+RDIMX*RDIMY-1)/(RDIMX*RDIMY))/RLOAD;
  parAlmond->rho  = (dfloat*) calloc(3*numBlocks,sizeof(dfloat));
  parAlmond->o_rho  = device.malloc(3*numBlocks*sizeof(dfloat), parAlmond->rho); 
}

void parAlmondReport(parAlmond_t *parAlmond) {

  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  if(rank==0) {
    printf("------------------ParAlmond Report-----------------------------------\n");
    printf("---------------------------------------------------------------------\n");
    printf("level| active ranks |   dimension   |  nnzs         |  nnz/row      |\n");
    printf("     |              | (min,max,avg) | (min,max,avg) | (min,max,avg) |\n");
    printf("---------------------------------------------------------------------\n");
  }

  for(int lev=0; lev<parAlmond->numLevels; lev++){

    dlong Nrows = parAlmond->levels[lev]->Nrows;
    hlong hNrows = (hlong) parAlmond->levels[lev]->Nrows;

    int active = (Nrows>0) ? 1:0;
    int totalActive=0;
    MPI_Allreduce(&active, &totalActive, 1, MPI_INT, MPI_SUM, agmg::comm);

    dlong minNrows=0, maxNrows=0;
    hlong totalNrows=0;
    dfloat avgNrows;
    MPI_Allreduce(&Nrows, &maxNrows, 1, MPI_DLONG, MPI_MAX, agmg::comm);
    MPI_Allreduce(&hNrows, &totalNrows, 1, MPI_HLONG, MPI_SUM, agmg::comm);
    avgNrows = (dfloat) totalNrows/totalActive;

    if (Nrows==0) Nrows=maxNrows; //set this so it's ignored for the global min
    MPI_Allreduce(&Nrows, &minNrows, 1, MPI_DLONG, MPI_MIN, agmg::comm);


    long long int nnz;
    if (parAlmond->levels[lev]->A)
      nnz = parAlmond->levels[lev]->A->diagNNZ+parAlmond->levels[lev]->A->offdNNZ;
    else
      nnz =0;
    long long int minNnz=0, maxNnz=0, totalNnz=0;
    dfloat avgNnz;
    MPI_Allreduce(&nnz, &maxNnz, 1, MPI_LONG_LONG_INT, MPI_MAX, agmg::comm);
    MPI_Allreduce(&nnz, &totalNnz, 1, MPI_LONG_LONG_INT, MPI_SUM, agmg::comm);
    avgNnz = (dfloat) totalNnz/totalActive;

    if (nnz==0) nnz = maxNnz; //set this so it's ignored for the global min
    MPI_Allreduce(&nnz, &minNnz, 1, MPI_LONG_LONG_INT, MPI_MIN, agmg::comm);

    Nrows = parAlmond->levels[lev]->Nrows;
    dfloat nnzPerRow = (Nrows==0) ? 0 : (dfloat) nnz/Nrows;
    dfloat minNnzPerRow=0, maxNnzPerRow=0, avgNnzPerRow=0;
    MPI_Allreduce(&nnzPerRow, &maxNnzPerRow, 1, MPI_DFLOAT, MPI_MAX, agmg::comm);
    MPI_Allreduce(&nnzPerRow, &avgNnzPerRow, 1, MPI_DFLOAT, MPI_SUM, agmg::comm);
    avgNnzPerRow /= totalActive;

    if (Nrows==0) nnzPerRow = maxNnzPerRow;
    MPI_Allreduce(&nnzPerRow, &minNnzPerRow, 1, MPI_DFLOAT, MPI_MIN, agmg::comm);

    if (rank==0){
      printf(" %3d |        %4d  |   %10.2f  |   %10.2f  |   %10.2f  |\n",
        lev, totalActive, (dfloat)minNrows, (dfloat)minNnz, minNnzPerRow);
      printf("     |              |   %10.2f  |   %10.2f  |   %10.2f  |\n",
        (dfloat)maxNrows, (dfloat)maxNnz, maxNnzPerRow);
      printf("     |              |   %10.2f  |   %10.2f  |   %10.2f  |\n",
        avgNrows, avgNnz, avgNnzPerRow);
    }
  }
  if(rank==0)
    printf("---------------------------------------------------------------------\n");
}
   

//create coarsened problem
void coarsenAgmgLevel(agmgLevel *level, csr **coarseA, csr **P, csr **R, dfloat **nullCoarseA, setupAide options){

  printf("\n >>>>  coarsenAGMGLevel is called !!!  \n");
  // establish the graph of strong connections
  int MyKey=0;
  options.getArgs("MYFUNCTIONS",MyKey);   // get from setup file
 
  csr *C; 
  hlong *FineToCoarse;	
  
  if (MyKey==0){
	    level->threshold = 0.5;                             // probably I need a different threshold (epsilon)  
	    options.getArgs("COARSENTHRESHOLD",level->threshold);   // get from setup file
		C = strong_graph(level->A, level->threshold);  // default so change it by the one that I need 
        FineToCoarse = form_aggregates(level, C);    // Here I will add smooth aggregation as new option 
        //find_aggregate_owners(level,FineToCoarse,options);  
        // *P = construct_interpolator(level, FineToCoarse, nullCoarseA);   	 	// construct the Prolongation... hopefully is the same	 
       
  }
  else if (MyKey ==1){
	    /*options.getArgs("COARSENTHRESHOLD",level->threshold);   // get from setup file
     	C = strong_graph2(level->A, level->threshold);   // my strong graph :)
		FineToCoarse = form_aggregates2(level, C);    // my algorithm		 
		//find_aggregate_owners(level,FineToCoarse,options);  // ?  */
		
		options.getArgs("COARSENTHRESHOLD",level->threshold);   // get from setup file
     	C = strong_graph(level->A, level->threshold);   // my strong graph :)
		FineToCoarse = form_aggregates3(level, C,options);    // my algorithm		 
		
		
	
  } else{
	    options.getArgs("COARSENTHRESHOLD",level->threshold);   // get from setup file
     	C = strong_graph(level->A, level->threshold);   // my strong graph :)
		FineToCoarse = form_aggregates2(level, C);    // my algorithm		 
 }

  find_aggregate_owners(level,FineToCoarse,options);  // ?
  
  /*
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;
   
  printf("\n===================== \n C (Rank = %d) diagNNZ =%d Nrows = %d Ncols = %d NlocalCols =  %d\n",rank,C->diagNNZ,C->Nrows, C->Ncols,C->NlocalCols);
  if (C->diagNNZ){
	for (int i = 0; i<C->Nrows ; i++){
		int Jstart = C->diagRowStarts[i], Jend = C->diagRowStarts[i+1];
		for(int j=Jstart;j<Jend;j++){
			printf("(%d,%d)  ",i,C->diagCols[j] );
		}
		printf("\t\t [%d,%d] \n",Jstart,Jend);
	}
  }
  
  printf("\n===================== \n C (Rank = %d) offdNNZ = %d Nrows = %d Ncols = %d NlocalCols =  %d\n",rank,C->offdNNZ, C->Nrows, C->Ncols,C->NlocalCols);
  if (C->offdNNZ){
  for (int i = 0; i<C->Nrows ; i++){
		int Jstart = C->offdRowStarts[i], Jend = C->offdRowStarts[i+1];
		for(int j=Jstart;j<Jend;j++){
			printf("(%d,%d)  ",i,C->offdCols[j] );
		}
		printf("\t\t [%d,%d] \n",Jstart,Jend);
	}
  }
  
  
  
  hlong *globalIndex = level->globalRowStarts;
  
  printf("\n==============\n global indices \n==================\n");
  
  printf("\n===================== \n C (Rank = %d) diagNNZ =%d Nrows = %d Ncols = %d NlocalCols =  %d   GlobalIndex= %d \n",rank,C->diagNNZ,C->Nrows, C->Ncols,C->NlocalCols,globalIndex[rank]);
  if (C->diagNNZ){
	for (int i = 0; i<C->Nrows ; i++){
		int Jstart = C->diagRowStarts[i], Jend = C->diagRowStarts[i+1];
		for(int j=Jstart;j<Jend;j++){
			printf("(%d,%d)  ",i+globalIndex[rank],level->A->colMap[C->diagCols[j]]);
		}
		printf("\t\t [%d,%d] \n",Jstart,Jend);
	}
  }
  
  printf("\n===================== \n C (Rank = %d) offdNNZ = %d Nrows = %d Ncols = %d NlocalCols =  %d    GlobalIndex= %d \n",rank,C->offdNNZ, C->Nrows, C->Ncols,C->NlocalCols,globalIndex[rank]);
  if (C->offdNNZ){
  for (int i = 0; i<C->Nrows ; i++){
		int Jstart = C->offdRowStarts[i], Jend = C->offdRowStarts[i+1];
		for(int j=Jstart;j<Jend;j++){
			printf("(%d,%d)  ",i+globalIndex[rank],level->A->colMap[C->diagCols[j]]);
		}
		printf("\t\t [%d,%d] \n",Jstart,Jend);
	}
  }
  */
  
  
  *P = construct_interpolator(level, FineToCoarse,nullCoarseA);   	 	// construct the Prolongation... hopefully is the same 
  
  
 /* printf("\n+++++\n FinetoCoarse \n");
  for (int i=0;i<level->A->Nrows;i++)
	printf("%d   ",FineToCoarse[i]);
  printf("\n+++++\n FinetoCoarse \n");
  
                                
  // smooth step
  // (I -  M^{-1} * A)*P
  // dumped Jacobi : M^{-1} = omega*D^{-1}
  //  *P = OneSmooth(w,A,P); 
  
 // for (int k =0; k<(*P)->Nrows+1;k++)
//			printf("%d ",(*P)->diagRowStarts[k]);
		
  /*
  printf("\n+++++++++++++++++\n print P \n ++++++++++++++++++++++++++\n ");			
       for(dlong i=0; i<(*P)->Nrows; i++){
			dlong Jstart = (*P)->diagRowStarts[i], Jend = (*P)->diagRowStarts[i+1]; // start & end of rows for "node i"
		//	printf("Jstart = %d \t Jend = %d",Jstart,Jend);
			for(dlong jj = Jstart; jj<Jend; jj++){
				printf("(%d,%d) = %.3f ",i,(*P)->diagCols[jj],(*P)->diagCoefs[jj]);
			}
			printf("\t(Fila %d) \n",i);
		}
		printf("\n+++++++++++++++++\n ");  */
		
	/*	
	printf("\n+++++++++++++++++\n print A indices\n ++++++++++++++++++++++++++\n ");
			
       for(dlong i=0; i<P->Nrows; i++){
			dlong Jstart = P->diagRowStarts[i], Jend = P->diagRowStarts[i+1]; // start & end of rows for "node i"
			for(dlong jj = Jstart; jj<Jend; jj++){
				printf("(%d,%d)  ",i,P->diagCols[jj]);
			}
			printf("\t\t[row %d] \n",i);
		}
		printf("\n+++++++++++++++++\n ");	
		
		
		 
  
	printf("\n+++++++++++++++++\n print A column indices\n ++++++++++++++++++++++++++\n ");
			
       for(dlong i=0; i<P->diagNNZ; i++){
			printf("(%d,%d)",i,P->diagCols[i]);
		}
		printf("\n+++++++++++++++++\n ");	
		
*/
		 
  
     
 //  *R = transpose(level,*P, level->globalRowStarts, level->globalAggStarts);	// R = P' so it should be the same 
   //OneSmooth(0.66,level->A,*R); 
  
  
  //for (int k =0; k<(*R)->Nrows+1;k++)
//			printf("%d ",(*R)->diagRowStarts[k]);
		
  /*
  printf("\n+++++++++++++++++\n print R \n ++++++++++++++++++++++++++\n ");
			
       for(dlong i=0; i<(*R)->Nrows; i++){
			dlong Jstart = (*R)->diagRowStarts[i], Jend = (*R)->diagRowStarts[i+1]; // start & end of rows for "node i"
		//	printf("Jstart = %d \t Jend = %d",Jstart,Jend);
			for(dlong jj = Jstart; jj<Jend; jj++){
				printf("(%d,%d) = %.3f ",i,(*R)->diagCols[jj],(*R)->diagCoefs[jj]);
			}
			printf("\t(Fila %d) \n",i);
		}
		printf("\n+++++++++++++++++\n ");*/
	
/*   *P = OneSmooth(level,0.66,level->A,*R);     
  printf("\n+++++++++++++++++\n print P OneSmooth   (%d,%d) \n ++++++++++++++++++++++++++\n ",(*P)->Nrows,(*P)->Ncols);
			
       for(dlong i=0; i<(*P)->Nrows; i++){
			dlong Jstart = (*P)->diagRowStarts[i], Jend = (*P)->diagRowStarts[i+1]; // start & end of rows for "node i"
			printf("Jstart = %d \t Jend = %d\n",Jstart,Jend);
			for(dlong jj = Jstart; jj<Jend; jj++){
				printf("(%d,%d,%d) = %.3f ",i,jj,(*P)->diagCols[jj],(*P)->diagCoefs[jj]);
			}
			printf("\t(Fila %d) \n",i);
		}
		printf("\n+++++++++++++++++\n ");
  
  
  */
  
  *R = transpose(level, *P, level->globalRowStarts, level->globalAggStarts);	// R = P' so it should be the same 
  
  *coarseA = galerkinProd(level, *R, level->A, *P);				// A_2h = P(A_h)R so it should be the same
   
}

csr * strong_graph(csr *A, dfloat threshold){
  const dlong N = A->Nrows;
  const dlong M = A->Ncols;




  csr *C = (csr *) calloc(1, sizeof(csr));   // allocate memory

  C->Nrows = N;  // set number of rows
  C->Ncols = M;  // set number of columns

  C->diagRowStarts = (dlong *) calloc(N+1,sizeof(dlong));    // allocate memory for local data
  C->offdRowStarts = (dlong *) calloc(N+1,sizeof(dlong));    // allocate memory for non-local data 

  dfloat *maxOD;
  if (N) maxOD = (dfloat *) calloc(N,sizeof(dfloat));  // create "maxOD" and set each entry to 0

  //store the diagonal of A for all needed columns
  dfloat *diagA = (dfloat *) calloc(M,sizeof(dfloat));
  for (dlong i=0;i<N;i++)
    diagA[i] = A->diagCoefs[A->diagRowStarts[i]];

  csrHaloExchange(A, sizeof(dfloat), diagA, A->sendBuffer, diagA+A->NlocalCols);

  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    dfloat sign = (diagA[i] >= 0) ? 1:-1;    // compute sign of A[i][i]
    dfloat Aii = fabs(diagA[i]);

    //find maxOD
    //local entries  // compute the max S_i*A_ij for each "node i" in local entries
    dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1];
    for(dlong jj= Jstart+1; jj<Jend; jj++){
      dlong col = A->diagCols[jj];
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->diagCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));
      if(OD > maxOD[i]) maxOD[i] = OD;  // stored here 
    }
    //non-local entries  // compute the max S_i * A_ij for each "node i" in non-local entries
    Jstart = A->offdRowStarts[i], Jend = A->offdRowStarts[i+1];
    for(dlong jj= Jstart; jj<Jend; jj++){
      dlong col = A->offdCols[jj];
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->offdCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));
      if(OD > maxOD[i]) maxOD[i] = OD; // if this is greater than the local one, it's updated
    }

    int diag_strong_per_row = 1; // diagonal entry
    //local entries
    Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1];
    for(dlong jj = Jstart+1; jj<Jend; jj++){
      dlong col = A->diagCols[jj];
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->diagCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));  // this is not the measure in the paper ( this is normalized )
      if(OD > threshold*maxOD[i]) diag_strong_per_row++;     // count the number of local strong connections in "column i"
    }
    int offd_strong_per_row = 0;
    //non-local entries
    Jstart = A->offdRowStarts[i], Jend = A->offdRowStarts[i+1];
    for(dlong jj= Jstart; jj<Jend; jj++){
      dlong col = A->offdCols[jj];
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->offdCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));
      if(OD > threshold*maxOD[i]) offd_strong_per_row++;    // count the number of non-local strong connections in "column i"
    }

    C->diagRowStarts[i+1] = diag_strong_per_row;    // store in "i+1" the number of strong connected local entries
    C->offdRowStarts[i+1] = offd_strong_per_row;    // store in "i+1" the number of strong connected non-local entries
  }

  // cumulative sum
  for(dlong i=1; i<N+1 ; i++) {
    C->diagRowStarts[i] += C->diagRowStarts[i-1]; // update diagRowStarts[i] = diagRowStarts[i-1] + number of entries
    C->offdRowStarts[i] += C->offdRowStarts[i-1]; // update  offRowStarts[i] = offRowStarts[i-1] + number of entries 
  }

  C->diagNNZ = C->diagRowStarts[N];   // update the size of local entries
  C->offdNNZ = C->offdRowStarts[N];   // update the size of non-local entries

  if (C->diagNNZ) C->diagCols = (dlong *) calloc(C->diagNNZ, sizeof(dlong));  // allocate memory for the local entries and set it to 0
  if (C->offdNNZ) C->offdCols = (dlong *) calloc(C->offdNNZ, sizeof(dlong));  // allocate memory for the non-local entries  and set it to 0

  // fill in the columns for strong connections
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    dfloat sign = (diagA[i] >= 0) ? 1:-1;    // compute sign of A[i][i]
    dfloat Aii = fabs(diagA[i]);	     // compute abs value of A[i][i]

    dlong diagCounter = C->diagRowStarts[i]; // start of the local entries in "column i"
    dlong offdCounter = C->offdRowStarts[i]; // start of the non-local entries in "column i"

    //local entries
    C->diagCols[diagCounter++] = i;  // diag entry  // for each "node i"
    dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1]; // start & end of rows for "node i"
    for(dlong jj = Jstart+1; jj<Jend; jj++){     // loop over all entries in (start/end)
      dlong col = A->diagCols[jj];               
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->diagCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));
      if(OD > threshold*maxOD[i])      // compare the measure of each entry  wrt ( threshold * max )
        C->diagCols[diagCounter++] = A->diagCols[jj];  //  store the corresponding columns for  indices
    }
    // non-local entries   // for each "node i"
    Jstart = A->offdRowStarts[i], Jend = A->offdRowStarts[i+1];   // start & end of rows for "node i"
    for(dlong jj = Jstart; jj<Jend; jj++){    // loop over all entries in (start/end)
      dlong col = A->offdCols[jj];
      dfloat Ajj = fabs(diagA[col]);
      dfloat OD = -sign*A->offdCoefs[jj]/(sqrt(Aii)*sqrt(Ajj));
      if(OD > threshold*maxOD[i])      // compare the measure of each entry  wrt ( threshold * max )
        C->offdCols[offdCounter++] = A->offdCols[jj];  //  store the corresponding columns for  indices
    }
  }
  if(N) free(maxOD);

  // this exploit the format CSR, since C is fake matrix since it only create the indices corresponding to the strong connections
  // but not the entry which is 1 and usesless. The strong connections are: 

   /*for (dlong i = 0 ; i < N : i++){
 	for (dlong jj = C-diagRowStarts[i]+1; jj < C->diagRowStarts[i+1] ; jj++)
	    printf("\n(%d,%d)\n", i , C->diagCols[jj] );
	for (dlong jj = C->offdRowStarts[i] ; jj < C->offdRowStarts[i+1] ; jj++)
	    printf("\n(%d,%d)\n", i , C->offdCols[jj] );
     }*/		  

  printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("\n+++\t Stron_graph :N=%d\tM=%d diagNNZ = %d \t  offdNNZ = %d \t   ++++++\n",N,M,C->diagNNZ,C->offdNNZ);
  printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");



     		 
  return C;
}

bool customLess(int smax, dfloat rmax, hlong imax, int s, dfloat r, hlong i){

  if(s > smax) return true;
  if(smax > s) return false;

  if(r > rmax) return true;
  if(rmax > r) return false;

  if(i > imax) return true;
  if(i < imax) return false;

  return false;
}



hlong * form_aggregates(agmgLevel *level, csr *C){

  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diagNNZ;
  const dlong offdNNZ = C->offdNNZ;
  
  hlong *FineToCoarse = (hlong *) calloc(M, sizeof(hlong));
  for (dlong i =0;i<M;i++) FineToCoarse[i] = -1;

  dfloat *rands  = (dfloat *) calloc(M, sizeof(dfloat));
  int   *states = (int *)   calloc(M, sizeof(int));

  dfloat *Tr = (dfloat *) calloc(M, sizeof(dfloat));
  int    *Ts = (int *)    calloc(M, sizeof(int));
  hlong  *Ti = (hlong *)  calloc(M, sizeof(hlong));
  hlong  *Tc = (hlong *)  calloc(M, sizeof(hlong));

  csr *A = level->A;
  hlong *globalRowStarts = level->globalRowStarts;

  int    *intSendBuffer;
  hlong  *hlongSendBuffer;
  dfloat *dfloatSendBuffer;
  if (level->A->NsendTotal) {
    intSendBuffer = (int *) calloc(A->NsendTotal,sizeof(int));
    hlongSendBuffer = (hlong *) calloc(A->NsendTotal,sizeof(hlong));
    dfloatSendBuffer = (dfloat *) calloc(A->NsendTotal,sizeof(dfloat));
  }

  for(dlong i=0; i<N; i++)
    rands[i] = (dfloat) drand48();

  for(dlong i=0; i<N; i++)
    states[i] = 0;

  // add the number of non-zeros in each column
  //local non-zeros
  for(dlong i=0; i<diagNNZ; i++){
	rands[C->diagCols[i]] += 1.;
  }
 

  int *nnzCnt, *recvNnzCnt;
  if (A->NHalo) nnzCnt = (int *) calloc(A->NHalo,sizeof(int));
  if (A->NsendTotal) recvNnzCnt = (int *) calloc(A->NsendTotal,sizeof(int));


  //count the non-local non-zeros
  for (dlong i=0;i<offdNNZ;i++){    
    nnzCnt[C->offdCols[i]-A->NlocalCols]++;
  }
 
  
  
  //do a reverse halo exchange
  int tag = 999;

  // initiate immediate send  and receives to each other process as needed
  dlong recvOffset = 0;
  dlong sendOffset = 0;
  int sendMessage = 0, recvMessage = 0;
  for(int r=0;r<size;++r){
    if (A->NsendTotal) {
      if(A->NsendPairs[r]) {
        MPI_Irecv(recvNnzCnt+sendOffset, A->NsendPairs[r], MPI_INT, r, tag,
            agmg::comm, (MPI_Request*)A->haloSendRequests+sendMessage);
        sendOffset += A->NsendPairs[r];
        ++sendMessage;
      }
    }
    if (A->NrecvTotal) {
      if(A->NrecvPairs[r]){
        MPI_Isend(nnzCnt+recvOffset, A->NrecvPairs[r], MPI_INT, r, tag,
            agmg::comm, (MPI_Request*)A->haloRecvRequests+recvMessage);
        recvOffset += A->NrecvPairs[r];
        ++recvMessage;
      }
    }
  }

  // Wait for all sent messages to have left and received messages to have arrived
  if (A->NrecvTotal) {
    MPI_Status *sendStatus = (MPI_Status*) calloc(A->NsendMessages, sizeof(MPI_Status));
    MPI_Waitall(A->NsendMessages, (MPI_Request*)A->haloSendRequests, sendStatus);
    free(sendStatus);
  }
  if (A->NsendTotal) {
    MPI_Status *recvStatus = (MPI_Status*) calloc(A->NrecvMessages, sizeof(MPI_Status));
    MPI_Waitall(A->NrecvMessages, (MPI_Request*)A->haloRecvRequests, recvStatus);
    free(recvStatus);
  }

  for(int i=0;i<A->NsendTotal;++i){
    // local index of outgoing element in halo exchange
    dlong id = A->haloElementList[i];

    rands[id] += recvNnzCnt[i];
  }

  if (A->NHalo) free(nnzCnt);
  if (A->NsendTotal) free(recvNnzCnt);

  //share randomizer values
  csrHaloExchange(A, sizeof(dfloat), rands, dfloatSendBuffer, rands+A->NlocalCols);



  hlong done = 0;
  while(!done){
    // first neighbours
    #pragma omp parallel for
    for(dlong i=0; i<N; i++){

      int smax = states[i];
      dfloat rmax = rands[i];
      hlong imax = i + globalRowStarts[rank];

      if(smax != 1){
        //local entries
        for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
          const dlong col = C->diagCols[jj];
          if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
            smax = states[col];
            rmax = rands[col];
            imax = col + globalRowStarts[rank];
          }
        }
        //nonlocal entries
        for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
          const dlong col = C->offdCols[jj];
          if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])) {
            smax = states[col];
            rmax = rands[col];
            imax = A->colMap[col];
          }
        }
      }
      Ts[i] = smax;
      Tr[i] = rmax;
      Ti[i] = imax;
    }

    //share results
    csrHaloExchange(A, sizeof(dfloat), Tr, dfloatSendBuffer, Tr+A->NlocalCols);
    csrHaloExchange(A, sizeof(int), Ts, intSendBuffer, Ts+A->NlocalCols);
    csrHaloExchange(A, sizeof(hlong), Ti, hlongSendBuffer, Ti+A->NlocalCols);

    // second neighbours
    #pragma omp parallel for
    for(dlong i=0; i<N; i++){
      int    smax = Ts[i];
      dfloat rmax = Tr[i];
      hlong  imax = Ti[i];

      //local entries
      for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
        const dlong col = C->diagCols[jj];
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
        const dlong col = C->offdCols[jj];
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }

      // if I am the strongest among all the 1 and 2 ring neighbours
      // I am an MIS node
      if((states[i] == 0) && (imax == (i + globalRowStarts[rank])))
        states[i] = 1;

      // if there is an MIS node within distance 2, I am removed
      if((states[i] == 0) && (smax == 1))
        states[i] = -1;
    }

    csrHaloExchange(A, sizeof(int), states, intSendBuffer, states+A->NlocalCols);

    // if number of undecided nodes = 0, algorithm terminates
    hlong cnt = std::count(states, states+N, 0);
    MPI_Allreduce(&cnt,&done,1,MPI_HLONG, MPI_SUM,agmg::comm);
    done = (done == 0) ? 1 : 0;
  }
  
   

  dlong numAggs = 0;
  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));
  level->globalAggStarts = (hlong *) calloc(size+1,sizeof(hlong));
  // count the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    if(states[i] == 1) numAggs++;

  MPI_Allgather(&numAggs,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,agmg::comm);

  level->globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    level->globalAggStarts[r+1] = level->globalAggStarts[r] + gNumAggs[r];

  numAggs = 0;
  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    if(states[i] == 1)
      FineToCoarse[i] = level->globalAggStarts[rank] + numAggs++;

  //share the initial aggregate flags
  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);

  // form the aggregates
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int   smax = states[i];
    dfloat rmax = rands[i];
    hlong  imax = i + globalRowStarts[rank];
    hlong  cmax = FineToCoarse[i];

    if(smax != 1){
      //local entries
      for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
        const dlong col = C->diagCols[jj];
        if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
          smax = states[col];
          rmax = rands[col];
          imax = col + globalRowStarts[rank];
          cmax = FineToCoarse[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
        const dlong col = C->offdCols[jj];
        if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])){
          smax = states[col];
          rmax = rands[col];
          imax = A->colMap[col];
          cmax = FineToCoarse[col];
        }
      }
    }
    Ts[i] = smax;
    Tr[i] = rmax;
    Ti[i] = imax;
    Tc[i] = cmax;

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }


  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);
  csrHaloExchange(A, sizeof(dfloat), Tr, dfloatSendBuffer, Tr+A->NlocalCols);
  csrHaloExchange(A, sizeof(int), Ts, intSendBuffer, Ts+A->NlocalCols);
  csrHaloExchange(A, sizeof(hlong), Ti, hlongSendBuffer, Ti+A->NlocalCols);
  csrHaloExchange(A, sizeof(hlong), Tc, hlongSendBuffer, Tc+A->NlocalCols);

  // second neighbours
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int    smax = Ts[i];
    dfloat rmax = Tr[i];
    hlong  imax = Ti[i];
    hlong  cmax = Tc[i];

    //local entries
    for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
      const dlong col = C->diagCols[jj];
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }
    //nonlocal entries
    for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
      const dlong col = C->offdCols[jj];
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }

  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);

  free(rands);
  free(states);
  free(Tr);
  free(Ts);
  free(Ti);
  free(Tc);
  if (level->A->NsendTotal) {
    free(intSendBuffer);
    free(hlongSendBuffer);
    free(dfloatSendBuffer);
  }

  //TODO maybe free C here?
  // print aggregates
  /*
  printf("\n++++++++++++++\n  N agg = %d  \n+++++++++++++++++++\n",numAggs);
  int N_agg[numAggs];
  for (int i=0;i<numAggs;i++){
	  N_agg[i]=0; 
  }for (int i=0;i<N;i++){
	  N_agg[FineToCoarse[i]]++;
  }for (int i=0;i<numAggs;i++){
	  printf("\n Agg = %d  => %d",i,N_agg[i]);
  }
  */
  
  

  

  return FineToCoarse;
}

typedef struct {

  dlong fineId;
  hlong coarseId;
  hlong newCoarseId;

  int originRank;
  int ownerRank;

} parallelAggregate_t;

int compareOwner(const void *a, const void *b){
  parallelAggregate_t *pa = (parallelAggregate_t *) a;
  parallelAggregate_t *pb = (parallelAggregate_t *) b;

  if (pa->ownerRank < pb->ownerRank) return -1;
  if (pa->ownerRank > pb->ownerRank) return +1;

  return 0;
};

int compareAgg(const void *a, const void *b){
  parallelAggregate_t *pa = (parallelAggregate_t *) a;
  parallelAggregate_t *pb = (parallelAggregate_t *) b;

  if (pa->coarseId < pb->coarseId) return -1;
  if (pa->coarseId > pb->coarseId) return +1;

  if (pa->originRank < pb->originRank) return -1;
  if (pa->originRank > pb->originRank) return +1;

  return 0;
};

int compareOrigin(const void *a, const void *b){
  parallelAggregate_t *pa = (parallelAggregate_t *) a;
  parallelAggregate_t *pb = (parallelAggregate_t *) b;

  if (pa->originRank < pb->originRank) return -1;
  if (pa->originRank > pb->originRank) return +1;

  return 0;
};

void find_aggregate_owners(agmgLevel *level, hlong* FineToCoarse, setupAide options) {
  // MPI info
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  dlong N = level->A->Nrows;

  //Need to establish 'ownership' of aggregates
  
  //Keep the current partitioning for STRONGNODES. 
  // The rank that had the strong node for each aggregate owns the aggregate
  if (options.compareArgs("PARALMOND PARTITION", "STRONGNODES")) return;

  //populate aggregate array
  hlong gNumAggs = level->globalAggStarts[size]; //total number of aggregates
  
  parallelAggregate_t *sendAggs;
  if (N) 
    sendAggs = (parallelAggregate_t *) calloc(N,sizeof(parallelAggregate_t));
  else 
    sendAggs = (parallelAggregate_t *) calloc(1,sizeof(parallelAggregate_t));

  for (dlong i=0;i<N;i++) {
    sendAggs[i].fineId = i;
    sendAggs[i].originRank = rank;

    sendAggs[i].coarseId = FineToCoarse[i];

    //set a temporary owner. Evenly distibute aggregates amoungst ranks
    sendAggs[i].ownerRank = (int) (FineToCoarse[i]*size)/gNumAggs;
  }

  // Make the MPI_PARALLEL_AGGREGATE data type
  MPI_Datatype MPI_PARALLEL_AGGREGATE;
  MPI_Datatype dtype[5] = {MPI_DLONG, MPI_HLONG, MPI_HLONG, MPI_INT, MPI_INT};
  int blength[5] = {1, 1, 1, 1, 1};
  MPI_Aint addr[5], displ[5];
  MPI_Get_address ( &(sendAggs[0]            ), addr+0);
  MPI_Get_address ( &(sendAggs[0].coarseId   ), addr+1);
  MPI_Get_address ( &(sendAggs[0].newCoarseId), addr+2);
  MPI_Get_address ( &(sendAggs[0].originRank ), addr+3);
  MPI_Get_address ( &(sendAggs[0].ownerRank  ), addr+4);
  displ[0] = 0;
  displ[1] = addr[1] - addr[0];
  displ[2] = addr[2] - addr[0];
  displ[3] = addr[3] - addr[0];
  displ[4] = addr[4] - addr[0];
  MPI_Type_create_struct (5, blength, displ, dtype, &MPI_PARALLEL_AGGREGATE);
  MPI_Type_commit (&MPI_PARALLEL_AGGREGATE);

  //sort by owning rank for all_reduce
  qsort(sendAggs, N, sizeof(parallelAggregate_t), compareOwner);

  int *sendCounts = (int *) calloc(size,sizeof(int));
  int *recvCounts = (int *) calloc(size,sizeof(int));
  int *sendOffsets = (int *) calloc(size+1,sizeof(int));
  int *recvOffsets = (int *) calloc(size+1,sizeof(int));

  for(dlong i=0;i<N;++i)
    sendCounts[sendAggs[i].ownerRank]++;

  // find how many nodes to expect (should use sparse version)
  MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, agmg::comm);

  // find send and recv offsets for gather
  dlong recvNtotal = 0;
  for(int r=0;r<size;++r){
    sendOffsets[r+1] = sendOffsets[r] + sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r] + recvCounts[r];
    recvNtotal += recvCounts[r];
  }
  parallelAggregate_t *recvAggs = (parallelAggregate_t *) calloc(recvNtotal,sizeof(parallelAggregate_t));

  MPI_Alltoallv(sendAggs, sendCounts, sendOffsets, MPI_PARALLEL_AGGREGATE,
                recvAggs, recvCounts, recvOffsets, MPI_PARALLEL_AGGREGATE,
                agmg::comm);

  //sort by coarse aggregate number, and then by original rank
  qsort(recvAggs, recvNtotal, sizeof(parallelAggregate_t), compareAgg);

  //count the number of unique aggregates here
  dlong NumUniqueAggs =0;
  if (recvNtotal) NumUniqueAggs++;
  for (dlong i=1;i<recvNtotal;i++)
    if(recvAggs[i].coarseId!=recvAggs[i-1].coarseId) NumUniqueAggs++;

  //get their locations in the array
  dlong *aggStarts;
  if (NumUniqueAggs)
    aggStarts = (dlong *) calloc(NumUniqueAggs+1,sizeof(dlong));
  dlong cnt = 1;
  for (dlong i=1;i<recvNtotal;i++)
    if(recvAggs[i].coarseId!=recvAggs[i-1].coarseId) aggStarts[cnt++] = i;
  aggStarts[NumUniqueAggs] = recvNtotal;


  if (options.compareArgs("PARALMOND PARTITION", "DISTRIBUTED")) { //rank that contributes most to the aggregate ownes it
    //use a random dfloat for each rank to break ties.
    dfloat rand = (dfloat) drand48();
    dfloat *gRands = (dfloat *) calloc(size,sizeof(dfloat));
    MPI_Allgather(&rand, 1, MPI_DFLOAT, gRands, 1, MPI_DFLOAT, agmg::comm);

    //determine the aggregates majority owner
    int *rankCounts = (int *) calloc(size,sizeof(int));
    for (dlong n=0;n<NumUniqueAggs;n++) {
      //populate randomizer
      for (int r=0;r<size;r++)
        rankCounts[r] = gRands[r];

      //count the number of contributions to the aggregate from the separate ranks
      for (dlong i=aggStarts[n];i<aggStarts[n+1];i++)
        rankCounts[recvAggs[i].originRank]++;

      //find which rank is contributing the most to this aggregate
      int ownerRank = 0;
      dfloat maxEntries = rankCounts[0];
      for (int r=1;r<size;r++) {
        if (rankCounts[r]>maxEntries) {
          ownerRank = r;
          maxEntries = rankCounts[r];
        }
      }

      //set this aggregate's owner
      for (dlong i=aggStarts[n];i<aggStarts[n+1];i++)
        recvAggs[i].ownerRank = ownerRank;
    }
    free(gRands); free(rankCounts);
  } else { //default SATURATE: always choose the lowest rank to own the aggregate
    for (dlong n=0;n<NumUniqueAggs;n++) {
      
      int minrank = size;

      //count the number of contributions to the aggregate from the separate ranks
      for (dlong i=aggStarts[n];i<aggStarts[n+1];i++){

        minrank = (recvAggs[i].originRank<minrank) ? recvAggs[i].originRank : minrank;
      }

      //set this aggregate's owner
      for (dlong i=aggStarts[n];i<aggStarts[n+1];i++)
        recvAggs[i].ownerRank = minrank;
    }
  }
  free(aggStarts);

  //sort by owning rank
  qsort(recvAggs, recvNtotal, sizeof(parallelAggregate_t), compareOwner);

  int *newSendCounts = (int *) calloc(size,sizeof(int));
  int *newRecvCounts = (int *) calloc(size,sizeof(int));
  int *newSendOffsets = (int *) calloc(size+1,sizeof(int));
  int *newRecvOffsets = (int *) calloc(size+1,sizeof(int));

  for(dlong i=0;i<recvNtotal;++i)
    newSendCounts[recvAggs[i].ownerRank]++;

  // find how many nodes to expect (should use sparse version)
  MPI_Alltoall(newSendCounts, 1, MPI_INT, newRecvCounts, 1, MPI_INT, agmg::comm);

  // find send and recv offsets for gather
  dlong newRecvNtotal = 0;
  for(int r=0;r<size;++r){
    newSendOffsets[r+1] = newSendOffsets[r] + newSendCounts[r];
    newRecvOffsets[r+1] = newRecvOffsets[r] + newRecvCounts[r];
    newRecvNtotal += newRecvCounts[r];
  }
  parallelAggregate_t *newRecvAggs = (parallelAggregate_t *) calloc(newRecvNtotal,sizeof(parallelAggregate_t));

  MPI_Alltoallv(   recvAggs, newSendCounts, newSendOffsets, MPI_PARALLEL_AGGREGATE,
                newRecvAggs, newRecvCounts, newRecvOffsets, MPI_PARALLEL_AGGREGATE,
                agmg::comm);

  //sort by coarse aggregate number, and then by original rank
  qsort(newRecvAggs, newRecvNtotal, sizeof(parallelAggregate_t), compareAgg);

  //count the number of unique aggregates this rank owns
  dlong numAggs = 0;
  if (newRecvNtotal) numAggs++;
  for (dlong i=1;i<newRecvNtotal;i++)
    if(newRecvAggs[i].coarseId!=newRecvAggs[i-1].coarseId) numAggs++;

  //determine a global numbering of the aggregates
  dlong *lNumAggs = (dlong*) calloc(size,sizeof(dlong));
  MPI_Allgather(&numAggs, 1, MPI_DLONG, lNumAggs, 1, MPI_INT, agmg::comm);

  level->globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    level->globalAggStarts[r+1] = level->globalAggStarts[r] + lNumAggs[r];

  //set the new global coarse index
  cnt = level->globalAggStarts[rank];
  if (newRecvNtotal) newRecvAggs[0].newCoarseId = cnt;
  for (dlong i=1;i<newRecvNtotal;i++) {
    if(newRecvAggs[i].coarseId!=newRecvAggs[i-1].coarseId) cnt++;

    newRecvAggs[i].newCoarseId = cnt;
  }

  //sort by owning rank
  qsort(newRecvAggs, newRecvNtotal, sizeof(parallelAggregate_t), compareOrigin);

  for(int r=0;r<size;r++) sendCounts[r] = 0;  
  for(int r=0;r<=size;r++) {
    sendOffsets[r] = 0;
    recvOffsets[r] = 0;
  }

  for(dlong i=0;i<newRecvNtotal;++i)
    sendCounts[newRecvAggs[i].originRank]++;

  // find how many nodes to expect (should use sparse version)
  MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, agmg::comm);

  // find send and recv offsets for gather
  recvNtotal = 0;
  for(int r=0;r<size;++r){
    sendOffsets[r+1] = sendOffsets[r] + sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r] + recvCounts[r];
    recvNtotal += recvCounts[r];
  }

  //send the aggregate data back
  MPI_Alltoallv(newRecvAggs, sendCounts, sendOffsets, MPI_PARALLEL_AGGREGATE,
                   sendAggs, recvCounts, recvOffsets, MPI_PARALLEL_AGGREGATE,
                agmg::comm);

  //clean up
  MPI_Barrier(agmg::comm);
  MPI_Type_free(&MPI_PARALLEL_AGGREGATE);

  free(recvAggs);
  free(sendCounts);  free(recvCounts);
  free(sendOffsets); free(recvOffsets);
  free(newRecvAggs);
  free(newSendCounts);  free(newRecvCounts);
  free(newSendOffsets); free(newRecvOffsets);

  //record the new FineToCoarse map
  for (dlong i=0;i<N;i++)
    FineToCoarse[sendAggs[i].fineId] = sendAggs[i].newCoarseId;

  free(sendAggs);
}

csr *construct_interpolator(agmgLevel *level, hlong *FineToCoarse, dfloat **nullCoarseA){
  // MPI info
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  const dlong N = level->A->Nrows;
  // const dlong M = level->A->Ncols;

  hlong *globalAggStarts = level->globalAggStarts;

  const hlong globalAggOffset = level->globalAggStarts[rank];
  const dlong NCoarse = (dlong) (globalAggStarts[rank+1]-globalAggStarts[rank]); //local num agg

  csr* P = (csr *) calloc(1, sizeof(csr));

  P->Nrows = N;
  P->Ncols = NCoarse;

  P->NlocalCols = NCoarse;
  P->NHalo = 0;

  P->diagRowStarts = (dlong *) calloc(N+1, sizeof(dlong));
  P->offdRowStarts = (dlong *) calloc(N+1, sizeof(dlong));

  // each row has exactly one nonzero per row
  P->diagNNZ =0;
  P->offdNNZ =0;
  for(dlong i=0; i<N; i++) {
    hlong col = FineToCoarse[i];
    if ((col>globalAggOffset-1)&&(col<globalAggOffset+NCoarse)) {
      P->diagNNZ++;
      P->diagRowStarts[i+1]++;
    } else {
      P->offdNNZ++;
      P->offdRowStarts[i+1]++;
    }
  }
  for(dlong i=0; i<N; i++) {
    P->diagRowStarts[i+1] += P->diagRowStarts[i];
    P->offdRowStarts[i+1] += P->offdRowStarts[i];
  }

  if (P->diagNNZ) {
    P->diagCols  = (dlong *)  calloc(P->diagNNZ, sizeof(dlong));
    P->diagCoefs = (dfloat *) calloc(P->diagNNZ, sizeof(dfloat));
  }
  hlong *offdCols;
  if (P->offdNNZ) {
    offdCols  = (hlong *)  calloc(P->offdNNZ, sizeof(hlong));
    P->offdCols  = (dlong *)  calloc(P->offdNNZ, sizeof(dlong));
    P->offdCoefs = (dfloat *) calloc(P->offdNNZ, sizeof(dfloat));
  }

  dlong diagCnt = 0;
  dlong offdCnt = 0;
  for(dlong i=0; i<N; i++) {
    hlong col = FineToCoarse[i];
    if ((col>globalAggStarts[rank]-1)&&(col<globalAggStarts[rank+1])) {
      P->diagCols[diagCnt] = (dlong) (col - globalAggOffset); //local index
      P->diagCoefs[diagCnt++] = level->A->null[i];
    } else {
      offdCols[offdCnt] = col;
      P->offdCoefs[offdCnt++] = level->A->null[i];
    }
  }

  //record global indexing of columns
  P->colMap = (hlong *)   calloc(P->Ncols, sizeof(hlong));
  for (dlong i=0;i<P->Ncols;i++)
    P->colMap[i] = i + globalAggOffset;

  if (P->offdNNZ) {
    //we now need to reorder the x vector for the halo, and shift the column indices
    hlong *col = (hlong *) calloc(P->offdNNZ,sizeof(hlong));
    for (dlong i=0;i<P->offdNNZ;i++)
      col[i] = offdCols[i]; //copy non-local column global ids

    //sort by global index
    std::sort(col,col+P->offdNNZ);

    //count unique non-local column ids
    P->NHalo = 0;
    for (dlong i=1;i<P->offdNNZ;i++)
      if (col[i]!=col[i-1])  col[++P->NHalo] = col[i];
    P->NHalo++; //number of unique columns

    P->Ncols += P->NHalo;

    //save global column ids in colMap
    P->colMap = (hlong *) realloc(P->colMap, P->Ncols*sizeof(hlong));
    for (dlong i=0; i<P->NHalo; i++)
      P->colMap[i+P->NlocalCols] = col[i];
    free(col);

    //shift the column indices to local indexing
    for (dlong i=0;i<P->offdNNZ;i++) {
      hlong gcol = offdCols[i];
      for (dlong m=P->NlocalCols;m<P->Ncols;m++) {
        if (gcol == P->colMap[m])
          P->offdCols[i] = m;
      }
    }
    free(offdCols);
  }

  csrHaloSetup(P,globalAggStarts);

  // normalize the columns of P
  *nullCoarseA = (dfloat *) calloc(P->Ncols,sizeof(dfloat));

  //add local nonzeros
  for(dlong i=0; i<P->diagNNZ; i++)
    (*nullCoarseA)[P->diagCols[i]] += P->diagCoefs[i] * P->diagCoefs[i];

  dfloat *nnzSum, *recvNnzSum;
  if (P->NHalo) nnzSum = (dfloat *) calloc(P->NHalo,sizeof(dfloat));
  if (P->NsendTotal) recvNnzSum = (dfloat *) calloc(P->NsendTotal,sizeof(dfloat));

  //add the non-local non-zeros
  for (dlong i=0;i<P->offdNNZ;i++)
    nnzSum[P->offdCols[i]-P->NlocalCols] += P->offdCoefs[i] * P->offdCoefs[i];

  //do a reverse halo exchange
  int tag = 999;

  // initiate immediate send  and receives to each other process as needed
  dlong recvOffset = 0;
  dlong sendOffset = 0;
  int sendMessage = 0, recvMessage = 0;
  for(int r=0;r<size;++r){
    if (P->NsendTotal) {
      if(P->NsendPairs[r]) {
        MPI_Irecv(recvNnzSum+sendOffset, P->NsendPairs[r], MPI_DFLOAT, r, tag,
            agmg::comm, (MPI_Request*)P->haloSendRequests+sendMessage);
        sendOffset += P->NsendPairs[r];
        ++sendMessage;
      }
    }
    if (P->NrecvTotal) {
      if(P->NrecvPairs[r]){
        MPI_Isend(nnzSum+recvOffset, P->NrecvPairs[r], MPI_DFLOAT, r, tag,
            agmg::comm, (MPI_Request*)P->haloRecvRequests+recvMessage);
        recvOffset += P->NrecvPairs[r];
        ++recvMessage;
      }
    }
  }

  // Wait for all sent messages to have left and received messages to have arrived
  if (P->NrecvTotal) {
    MPI_Status *sendStatus = (MPI_Status*) calloc(P->NsendMessages, sizeof(MPI_Status));
    MPI_Waitall(P->NsendMessages, (MPI_Request*)P->haloSendRequests, sendStatus);
    free(sendStatus);
  }
  if (P->NsendTotal) {
    MPI_Status *recvStatus = (MPI_Status*) calloc(P->NrecvMessages, sizeof(MPI_Status));
    MPI_Waitall(P->NrecvMessages, (MPI_Request*)P->haloRecvRequests, recvStatus);
    free(recvStatus);
  }

  for(dlong i=0;i<P->NsendTotal;++i){
    // local index of outgoing element in halo exchange
    dlong id = P->haloElementList[i];

    (*nullCoarseA)[id] += recvNnzSum[i];
  }

  if (P->NHalo) free(nnzSum);

  for(dlong i=0; i<NCoarse; i++)
    (*nullCoarseA)[i] = sqrt((*nullCoarseA)[i]);

  csrHaloExchange(P, sizeof(dfloat), *nullCoarseA, P->sendBuffer, *nullCoarseA+P->NlocalCols);

  for(dlong i=0; i<P->diagNNZ; i++)
    P->diagCoefs[i] /= (*nullCoarseA)[P->diagCols[i]];
  for(dlong i=0; i<P->offdNNZ; i++)
    P->offdCoefs[i] /= (*nullCoarseA)[P->offdCols[i]];

  MPI_Barrier(agmg::comm);
  if (P->NsendTotal) free(recvNnzSum);

  return P;
}

typedef struct {

  hlong row;
  hlong col;
  dfloat val;
  int owner;

} nonzero_t;

int compareNonZero(const void *a, const void *b){
  nonzero_t *pa = (nonzero_t *) a;
  nonzero_t *pb = (nonzero_t *) b;

  if (pa->owner < pb->owner) return -1;
  if (pa->owner > pb->owner) return +1;

  if (pa->row < pb->row) return -1;
  if (pa->row > pb->row) return +1;

  if (pa->col < pb->col) return -1;
  if (pa->col > pb->col) return +1;

  return 0;
};

csr * transpose(agmgLevel* level, csr *A, hlong *globalRowStarts, hlong *globalColStarts){

  printf("\n++++++++++++++++++++++++\n Transpose called \n+++++++++++++++++++++++\n");

  // MPI info
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  csr *At = (csr *) calloc(1,sizeof(csr));

  At->Nrows = A->Ncols-A->NHalo;
  At->Ncols = A->Nrows;
  At->diagNNZ   = A->diagNNZ; //local entries remain local
  At->NlocalCols = At->Ncols;
  
  
  printf("At->Nrows = %d ;  At->Ncols = %d ;  At->diagNNZ   = %d ; At->NlocalCols = %d \n\n",At->Nrows,At->Ncols,At->diagNNZ,At->NlocalCols);
  

  At->diagRowStarts = (dlong *)   calloc(At->Nrows+1, sizeof(dlong));
  At->offdRowStarts = (dlong *)   calloc(At->Nrows+1, sizeof(dlong));

  //start with local entries
  if (A->diagNNZ) {
    At->diagCols      = (dlong *)  calloc(At->diagNNZ, sizeof(dlong));
    At->diagCoefs     = (dfloat *) calloc(At->diagNNZ, sizeof(dfloat));
  }

  // count the num of nonzeros per row for transpose
  for(dlong i=0; i<A->diagNNZ; i++){
    dlong row = A->diagCols[i];
    At->diagRowStarts[row+1]++;
  }

  // cumulative sum for rows
  for(dlong i=1; i<=At->Nrows; i++)
    At->diagRowStarts[i] += At->diagRowStarts[i-1];

  int *counter = (int *) calloc(At->Nrows+1,sizeof(int));
  for (dlong i=0; i<At->Nrows+1; i++)
    counter[i] = At->diagRowStarts[i];

  for(dlong i=0; i<A->Nrows; i++){
    const dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1];

    for(dlong jj=Jstart; jj<Jend; jj++){
      dlong row = A->diagCols[jj];
      At->diagCols[counter[row]]  = i;
      At->diagCoefs[counter[row]] = A->diagCoefs[jj];

      counter[row]++;
    }
  }
  free(counter);

  //record global indexing of columns
  At->colMap = (hlong *)   calloc(At->Ncols, sizeof(hlong));
  for (dlong i=0;i<At->Ncols;i++)
    At->colMap[i] = i + globalRowStarts[rank];

  //now the nonlocal entries. Need to reverse the halo exchange to send the nonzeros
  int tag = 999;

  nonzero_t *sendNonZeros;
  if (A->offdNNZ)
    sendNonZeros = (nonzero_t *) calloc(A->offdNNZ,sizeof(nonzero_t));

  int *Nsend = (int*) calloc(size, sizeof(int));
  int *Nrecv = (int*) calloc(size, sizeof(int));

  for(int r=0;r<size;r++) {
    Nsend[r] =0;
    Nrecv[r] =0;
  }

  // copy data from nonlocal entries into send buffer
  if (A->offdNNZ){  // when ther is offd data
	for(dlong i=0;i<A->Nrows;++i){
		for (dlong j=A->offdRowStarts[i];j<A->offdRowStarts[i+1];j++) {
			hlong col =  A->colMap[A->offdCols[j]]; //global ids
			for (int r=0;r<size;r++) { //find owner's rank
				if ((globalColStarts[r]-1<col) && (col < globalColStarts[r+1])) {
					Nsend[r]++;
					sendNonZeros[j].owner = r;
				}
			}
			sendNonZeros[j].row = col;
			sendNonZeros[j].col = i + globalRowStarts[rank];     //global ids
			sendNonZeros[j].val = A->offdCoefs[j];
		}
	}
  }
  //sort outgoing nonzeros by owner, then row and col
  if (A->offdNNZ)
    qsort(sendNonZeros, A->offdNNZ, sizeof(nonzero_t), compareNonZero);

  MPI_Alltoall(Nsend, 1, MPI_INT, Nrecv, 1, MPI_INT, agmg::comm);

  //count incoming nonzeros
  At->offdNNZ = 0;
  for (int r=0;r<size;r++)
    At->offdNNZ += Nrecv[r];

  nonzero_t *recvNonZeros;
  if (At->offdNNZ)
    recvNonZeros = (nonzero_t *) calloc(At->offdNNZ,sizeof(nonzero_t));

  // initiate immediate send and receives to each other process as needed
  int recvOffset = 0;
  int sendOffset = 0;
  int sendMessage = 0, recvMessage = 0;
  for(int r=0;r<size;++r){
    if (At->offdNNZ) {
      if(Nrecv[r]) {
        MPI_Irecv(((char*)recvNonZeros)+recvOffset, Nrecv[r]*sizeof(nonzero_t),
                      MPI_CHAR, r, tag, agmg::comm,
                      (MPI_Request*)A->haloSendRequests+recvMessage);
        recvOffset += Nrecv[r]*sizeof(nonzero_t);
        ++recvMessage;
      }
    }
    if (A->offdNNZ) {
      if(Nsend[r]){
        MPI_Isend(((char*)sendNonZeros)+sendOffset, Nsend[r]*sizeof(nonzero_t),
                      MPI_CHAR, r, tag, agmg::comm,
                      (MPI_Request*)A->haloRecvRequests+sendMessage);
        sendOffset += Nsend[r]*sizeof(nonzero_t);
        ++sendMessage;
      }
    }
  }

  // Wait for all sent messages to have left and received messages to have arrived
  if (A->offdNNZ) {
    MPI_Status *sendStatus = (MPI_Status*) calloc(sendMessage, sizeof(MPI_Status));
    MPI_Waitall(sendMessage, (MPI_Request*)A->haloRecvRequests, sendStatus);
    free(sendStatus);
  }
  if (At->offdNNZ) {
    MPI_Status *recvStatus = (MPI_Status*) calloc(recvMessage, sizeof(MPI_Status));
    MPI_Waitall(recvMessage, (MPI_Request*)A->haloSendRequests, recvStatus);
    free(recvStatus);
  }
  if (A->offdNNZ) free(sendNonZeros);

  //free(Nsend); free(Nrecv);

  if (At->offdNNZ) {
    //sort recieved nonzeros by row and col
    qsort(recvNonZeros, At->offdNNZ, sizeof(nonzero_t), compareNonZero);

    hlong *offdCols  = (hlong *)   calloc(At->offdNNZ,sizeof(hlong));
    At->offdCols  = (dlong *)   calloc(At->offdNNZ,sizeof(dlong));
    At->offdCoefs = (dfloat *) calloc(At->offdNNZ, sizeof(dfloat));

    //find row starts
    for(dlong n=0;n<At->offdNNZ;++n) {
      dlong row = (dlong) (recvNonZeros[n].row - globalColStarts[rank]);
      At->offdRowStarts[row+1]++;
    }
    //cumulative sum
    for (dlong i=0;i<At->Nrows;i++)
      At->offdRowStarts[i+1] += At->offdRowStarts[i];

    //fill cols and coefs
    for (dlong i=0; i<At->Nrows; i++) {
      for (dlong j=At->offdRowStarts[i]; j<At->offdRowStarts[i+1]; j++) {
        offdCols[j]  = recvNonZeros[j].col;
        At->offdCoefs[j] = recvNonZeros[j].val;
      }
    }
    free(recvNonZeros);

    //we now need to reorder the x vector for the halo, and shift the column indices
    hlong *col = (hlong *) calloc(At->offdNNZ,sizeof(hlong));
    for (dlong n=0;n<At->offdNNZ;n++)
      col[n] = offdCols[n]; //copy non-local column global ids

    //sort by global index
    std::sort(col,col+At->offdNNZ);

    //count unique non-local column ids
    At->NHalo = 0;
    for (dlong n=1;n<At->offdNNZ;n++)
      if (col[n]!=col[n-1])  col[++At->NHalo] = col[n];
    At->NHalo++; //number of unique columns

    At->Ncols += At->NHalo;

    //save global column ids in colMap
    At->colMap = (hlong *) realloc(At->colMap,At->Ncols*sizeof(hlong));
    for (dlong n=0; n<At->NHalo; n++)
      At->colMap[n+At->NlocalCols] = col[n];
    free(col);

    //shift the column indices to local indexing
    for (dlong n=0;n<At->offdNNZ;n++) {
      hlong gcol = offdCols[n];
      for (dlong m=At->NlocalCols;m<At->Ncols;m++) {
        if (gcol == At->colMap[m])
          At->offdCols[n] = m;
      }
    }
    free(offdCols);
  }

  csrHaloSetup(At,globalRowStarts);

  return At;
}

typedef struct {

  hlong coarseId;
  dfloat coef;

} pEntry_t;

typedef struct {

  hlong I;
  hlong J;
  dfloat coef;

} rapEntry_t;

int compareRAPEntries(const void *a, const void *b){
  rapEntry_t *pa = (rapEntry_t *) a;
  rapEntry_t *pb = (rapEntry_t *) b;

  if (pa->I < pb->I) return -1;
  if (pa->I > pb->I) return +1;

  if (pa->J < pb->J) return -1;
  if (pa->J > pb->J) return +1;

  return 0;
};

csr *galerkinProd(agmgLevel *level, csr *R, csr *A, csr *P){
	
	printf("\n++++++++++++++++++++++++\n Galerking product \n+++++++++++++++++++++++\n");
	printf("\n++++++++++++++++++++++++\n (%d,%d)(%d,%d)(%d,%d)   A->NsendTotal = %d \n+++++++++++++++++++++++\n",R->Nrows,R->Ncols,A->Nrows,A->Ncols,P->Nrows,P->Ncols,A->NsendTotal);
	

  // MPI info
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  hlong *globalAggStarts = level->globalAggStarts;
  // hlong *globalRowStarts = level->globalRowStarts;

  hlong globalAggOffset = globalAggStarts[rank];

  //The galerkin product can be computed as
  // (RAP)_IJ = sum_{i in Agg_I} sum_{j in Agg_j} P_iI A_ij P_jJ
  // Since each row of P has only one entry, we can share the ncessary
  // P entries, form the products, and send them to their destination rank

  dlong N = A->Nrows;
  dlong M = A->Ncols;

  printf("Level has %d rows, and is making %d aggregates\n", N, globalAggStarts[rank+1]-globalAggStarts[rank]);

  pEntry_t *PEntries;
  if (M) 
    PEntries = (pEntry_t *) calloc(M,sizeof(pEntry_t));
  else 
    PEntries = (pEntry_t *) calloc(1,sizeof(pEntry_t));

  //record the entries of P that this rank has
  //printf("\n++========================++\n P->diagNNz =%d   P->offdNNZ = %d \n\n", P->diagNNZ,P->offdNNZ) ;
  
  dlong cnt =0;
  for (dlong i=0;i<N;i++) {
	//printf("\n %d to %d (from 0 to %d) ",P->diagRowStarts[i],P->diagRowStarts[i+1],N);
    for (dlong j=P->diagRowStarts[i];j<P->diagRowStarts[i+1];j++) {
      PEntries[cnt].coarseId = P->diagCols[j] + globalAggOffset; //global ID
      PEntries[cnt].coef = P->diagCoefs[j];
      cnt++;
    }
    //printf("\t cnt = %d  ",cnt);
    //if (P->offdNNZ){
		//printf("\n no se debe ejecutar \n");
		for (dlong j=P->offdRowStarts[i];j<P->offdRowStarts[i+1];j++) {
		  PEntries[cnt].coarseId = P->colMap[P->offdCols[j]]; //global ID
		  PEntries[cnt].coef = P->offdCoefs[j];
		  cnt++;
		}
	//}
	//printf("\t sanity check cnt = %d and i = %d ",cnt,i);
 }


//printf("\n++========================++\n A->NsendTotal= %d \n\n", A->NsendTotal) ;	

  pEntry_t *entrySendBuffer;
  if (A->NsendTotal)
    entrySendBuffer = (pEntry_t *) calloc(A->NsendTotal,sizeof(pEntry_t));

  //fill in the entires of P needed in the halo
  csrHaloExchange(A, sizeof(pEntry_t), PEntries, entrySendBuffer, PEntries+A->NlocalCols);
  if (A->NsendTotal) free(entrySendBuffer);

  rapEntry_t *RAPEntries;
  dlong totalNNZ = A->diagNNZ+A->offdNNZ;
  if (totalNNZ) 
    RAPEntries = (rapEntry_t *) calloc(totalNNZ,sizeof(rapEntry_t));
  else 
    RAPEntries = (rapEntry_t *) calloc(1,sizeof(rapEntry_t)); //MPI_AlltoAll doesnt like null pointers
  
  // Make the MPI_RAPENTRY_T data type
  MPI_Datatype MPI_RAPENTRY_T;
  MPI_Datatype dtype[3] = {MPI_HLONG, MPI_HLONG, MPI_DFLOAT};
  int blength[3] = {1, 1, 1};
  MPI_Aint addr[3], displ[3];
  MPI_Get_address ( &(RAPEntries[0]     ), addr+0);
  MPI_Get_address ( &(RAPEntries[0].J   ), addr+1);
  MPI_Get_address ( &(RAPEntries[0].coef), addr+2);
  displ[0] = 0;
  displ[1] = addr[1] - addr[0];
  displ[2] = addr[2] - addr[0];
  MPI_Type_create_struct (3, blength, displ, dtype, &MPI_RAPENTRY_T);
  MPI_Type_commit (&MPI_RAPENTRY_T);

  //for the RAP products
  cnt =0;
  for (dlong i=0;i<N;i++) {
    for (dlong j=A->diagRowStarts[i];j<A->diagRowStarts[i+1];j++) {
      dlong col  = A->diagCols[j];
      dfloat coef = A->diagCoefs[j];

      RAPEntries[cnt].I = PEntries[i].coarseId;
      RAPEntries[cnt].J = PEntries[col].coarseId;
      RAPEntries[cnt].coef = coef*PEntries[i].coef*PEntries[col].coef;
      cnt++;
    }
  }
  //if (A->offdNNZ){
	  for (dlong i=0;i<N;i++) {
		for (dlong j=A->offdRowStarts[i];j<A->offdRowStarts[i+1];j++) {
		  dlong col  = A->offdCols[j];
		  dfloat coef = A->offdCoefs[j];

		  RAPEntries[cnt].I = PEntries[i].coarseId;
		  RAPEntries[cnt].J = PEntries[col].coarseId;
		  RAPEntries[cnt].coef = PEntries[i].coef*coef*PEntries[col].coef;
		  cnt++;
		}
	//}
  }
  
  //printf("\n++========================++\n A->diagNNz =%d   A->offdNNZ = %d \n\n", A->diagNNZ,A->offdNNZ) ;

  //sort entries by the coarse row and col
  if (totalNNZ) qsort(RAPEntries, totalNNZ, sizeof(rapEntry_t), compareRAPEntries);

  int *sendCounts = (int *) calloc(size,sizeof(int));
  int *recvCounts = (int *) calloc(size,sizeof(int));
  int *sendOffsets = (int *) calloc(size+1,sizeof(int));
  int *recvOffsets = (int *) calloc(size+1,sizeof(int));

  for(dlong i=0;i<totalNNZ;++i) {
    hlong id = RAPEntries[i].I;
    for (int r=0;r<size;r++) {
      if (globalAggStarts[r]-1<id && id < globalAggStarts[r+1])
        sendCounts[r]++;
    }
  }

  // find how many nodes to expect (should use sparse version)
  MPI_Alltoall(sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, agmg::comm);
  
  //printf("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n");

  // find send and recv offsets for gather
  dlong recvNtotal = 0;
  for(int r=0;r<size;++r){
    sendOffsets[r+1] = sendOffsets[r] + sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r] + recvCounts[r];
    recvNtotal += recvCounts[r];
  }
  rapEntry_t *recvRAPEntries;
  if (recvNtotal) 
    recvRAPEntries = (rapEntry_t *) calloc(recvNtotal,sizeof(rapEntry_t));
  else 
    recvRAPEntries = (rapEntry_t *) calloc(1,sizeof(rapEntry_t));//MPI_AlltoAll doesnt like null pointers
  
  MPI_Alltoallv(    RAPEntries, sendCounts, sendOffsets, MPI_RAPENTRY_T,
                recvRAPEntries, recvCounts, recvOffsets, MPI_RAPENTRY_T,
                agmg::comm);

  //sort entries by the coarse row and col
  if (recvNtotal) qsort(recvRAPEntries, recvNtotal, sizeof(rapEntry_t), compareRAPEntries);

  //count total number of nonzeros;
  dlong nnz =0;
  if (recvNtotal) nnz++;
  for (dlong i=1;i<recvNtotal;i++)
    if ((recvRAPEntries[i].I!=recvRAPEntries[i-1].I)||
          (recvRAPEntries[i].J!=recvRAPEntries[i-1].J)) nnz++;

  rapEntry_t *newRAPEntries;
  if (nnz)
    newRAPEntries = (rapEntry_t *) calloc(nnz,sizeof(rapEntry_t));
  else 
    newRAPEntries = (rapEntry_t *) calloc(1,sizeof(rapEntry_t));
  
  //compress nonzeros
  nnz = 0;
  if (recvNtotal) newRAPEntries[nnz++] = recvRAPEntries[0];
  for (dlong i=1;i<recvNtotal;i++) {
    if ((recvRAPEntries[i].I!=recvRAPEntries[i-1].I)||
          (recvRAPEntries[i].J!=recvRAPEntries[i-1].J)) {
      newRAPEntries[nnz++] = recvRAPEntries[i];
    } else {
      newRAPEntries[nnz-1].coef += recvRAPEntries[i].coef;
    }
  }

  dlong numAggs = (dlong) (globalAggStarts[rank+1]-globalAggStarts[rank]); //local number of aggregates

  csr *RAP = (csr*) calloc(1,sizeof(csr));

  RAP->Nrows = numAggs;
  RAP->Ncols = numAggs;

  RAP->NlocalCols = numAggs;

  RAP->diagRowStarts = (dlong *) calloc(numAggs+1, sizeof(dlong));
  RAP->offdRowStarts = (dlong *) calloc(numAggs+1, sizeof(dlong));

  for (dlong n=0;n<nnz;n++) {
    dlong row = (dlong) (newRAPEntries[n].I - globalAggOffset);
    if ((newRAPEntries[n].J > globalAggStarts[rank]-1)&&
          (newRAPEntries[n].J < globalAggStarts[rank+1])) {
      RAP->diagRowStarts[row+1]++;
    } else {
      RAP->offdRowStarts[row+1]++;
    }
  }

  // cumulative sum
  for(dlong i=0; i<numAggs; i++) {
    RAP->diagRowStarts[i+1] += RAP->diagRowStarts[i];
    RAP->offdRowStarts[i+1] += RAP->offdRowStarts[i];
  }
  RAP->diagNNZ = RAP->diagRowStarts[numAggs];
  RAP->offdNNZ = RAP->offdRowStarts[numAggs];

  printf("\n==================\n RAP->diagNNZ = %d    RAP->offdNNZ= %d \n",RAP->diagNNZ,RAP->offdNNZ)	;


  dlong *diagCols;
  dfloat *diagCoefs;
  if (RAP->diagNNZ) {
    RAP->diagCols  = (dlong *)   calloc(RAP->diagNNZ, sizeof(dlong));
    RAP->diagCoefs = (dfloat *) calloc(RAP->diagNNZ, sizeof(dfloat));
    diagCols  = (dlong *)   calloc(RAP->diagNNZ, sizeof(dlong));
    diagCoefs = (dfloat *) calloc(RAP->diagNNZ, sizeof(dfloat));
  }
  hlong *offdCols;
  if (RAP->offdNNZ) {
    offdCols  = (hlong *)   calloc(RAP->offdNNZ,sizeof(hlong));
    RAP->offdCols  = (dlong *)   calloc(RAP->offdNNZ,sizeof(dlong));
    RAP->offdCoefs = (dfloat *) calloc(RAP->offdNNZ, sizeof(dfloat));
  }

  dlong diagCnt =0;
  dlong offdCnt =0;
  for (dlong n=0;n<nnz;n++) {
    if ((newRAPEntries[n].J > globalAggStarts[rank]-1)&&
          (newRAPEntries[n].J < globalAggStarts[rank+1])) {
      diagCols[diagCnt]  = (dlong) (newRAPEntries[n].J - globalAggOffset);
      diagCoefs[diagCnt] = newRAPEntries[n].coef;
      diagCnt++;
    } else {
      offdCols[offdCnt]  = newRAPEntries[n].J;
      RAP->offdCoefs[offdCnt] = newRAPEntries[n].coef;
      offdCnt++;
    }
  }

  //move diagonal entries first
  for (dlong i=0;i<RAP->Nrows;i++) {
    dlong start = RAP->diagRowStarts[i];
    int cnt = 1;
    for (dlong j=RAP->diagRowStarts[i]; j<RAP->diagRowStarts[i+1]; j++) {
      if (diagCols[j] == i) { //move diagonal to first entry
        RAP->diagCols[start] = diagCols[j];
        RAP->diagCoefs[start] = diagCoefs[j];
      } else {
        RAP->diagCols[start+cnt] = diagCols[j];
        RAP->diagCoefs[start+cnt] = diagCoefs[j];
        cnt++;
      }
    }
  }

  //record global indexing of columns
  RAP->colMap = (hlong *)   calloc(RAP->Ncols, sizeof(hlong));
  for (dlong i=0;i<RAP->Ncols;i++)
    RAP->colMap[i] = i + globalAggOffset;

  if (RAP->offdNNZ) {
    //we now need to reorder the x vector for the halo, and shift the column indices
    hlong *col = (hlong *) calloc(RAP->offdNNZ,sizeof(hlong));
    for (dlong n=0;n<RAP->offdNNZ;n++)
      col[n] = offdCols[n]; //copy non-local column global ids

    //sort by global index
    std::sort(col,col+RAP->offdNNZ);

    //count unique non-local column ids
    RAP->NHalo = 0;
    for (dlong n=1;n<RAP->offdNNZ;n++)
      if (col[n]!=col[n-1])  col[++RAP->NHalo] = col[n];
    RAP->NHalo++; //number of unique columns

    RAP->Ncols += RAP->NHalo;

    //save global column ids in colMap
    RAP->colMap = (hlong *) realloc(RAP->colMap,RAP->Ncols*sizeof(hlong));
    for (dlong n=0; n<RAP->NHalo; n++)
      RAP->colMap[n+RAP->NlocalCols] = col[n];

    //shift the column indices to local indexing
    for (dlong n=0;n<RAP->offdNNZ;n++) {
      hlong gcol = offdCols[n];
      for (dlong m=RAP->NlocalCols;m<RAP->Ncols;m++) {
        if (gcol == RAP->colMap[m])
          RAP->offdCols[n] = m;
      }
    }
    free(col);
    free(offdCols);
  }
  csrHaloSetup(RAP,globalAggStarts);

  //clean up
  MPI_Barrier(agmg::comm);
  MPI_Type_free(&MPI_RAPENTRY_T);

  free(PEntries);
  free(sendCounts); free(recvCounts);
  free(sendOffsets); free(recvOffsets);
  if (RAP->diagNNZ) {
    free(diagCols);
    free(diagCoefs);
  }
  free(RAPEntries);
  free(newRAPEntries);
  free(recvRAPEntries);

  return RAP;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// My modifications
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

csr * strong_graph2(csr *A, dfloat threshold){

  const dlong N = A->Nrows;
  const dlong M = A->Ncols;

  csr *C = (csr *) calloc(1, sizeof(csr));   // allocate memory

  C->Nrows = N;  // set number of rows
  C->Ncols = M;  // set number of columns

  C->diagRowStarts = (dlong *) calloc(N+1,sizeof(dlong));    // allocate memory for local data
  C->offdRowStarts = (dlong *) calloc(N+1,sizeof(dlong));    // allocate memory for non-local data 

  dfloat *maxOD;
  if (N) maxOD = (dfloat *) calloc(N,sizeof(dfloat));  // create "maxOD" and set each entry to 0

  //store the diagonal of A for all needed columns
  dfloat *diagA = (dfloat *) calloc(M,sizeof(dfloat));
  for (dlong i=0;i<N;i++)
    diagA[i] = A->diagCoefs[A->diagRowStarts[i]];

  csrHaloExchange(A, sizeof(dfloat), diagA, A->sendBuffer, diagA+A->NlocalCols);

  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    //dfloat sign = (diagA[i] >= 0) ? 1:-1;    // compute sign of A[i][i]
    dfloat Aii = diagA[i];

    int diag_strong_per_row = 1; // diagonal entry
    //local entries
    dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1]; // start & end of rows for "node i"
    //Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1];
    for(dlong jj = Jstart+1; jj<Jend; jj++){
      dlong col = A->diagCols[jj];
      dfloat Ajj = diagA[col];
      if(fabs(Ajj) > threshold*sqrt(fabs(Ajj*Aii))) diag_strong_per_row++;     // count the number of local strong connections in "column i"
    }
    int offd_strong_per_row = 0;
    //non-local entries
    Jstart = A->offdRowStarts[i], Jend = A->offdRowStarts[i+1];
    for(dlong jj= Jstart; jj<Jend; jj++){
      dlong col = A->offdCols[jj];
      dfloat Ajj = diagA[col];
      if(fabs(Ajj) > threshold*sqrt(fabs(Ajj*Aii))) offd_strong_per_row++;    // count the number of non-local strong connections in "column i"
    }

    C->diagRowStarts[i+1] = diag_strong_per_row;    // store in "i+1" the number of strong connected local entries
    C->offdRowStarts[i+1] = offd_strong_per_row;    // store in "i+1" the number of strong connected non-local entries
  }

  // cumulative sum
  for(dlong i=1; i<N+1 ; i++) {
    C->diagRowStarts[i] += C->diagRowStarts[i-1]; // update diagRowStarts[i] = diagRowStarts[i-1] + number of entries
    C->offdRowStarts[i] += C->offdRowStarts[i-1]; // update  offRowStarts[i] = offRowStarts[i-1] + number of entries 
  }

  C->diagNNZ = C->diagRowStarts[N];   // update the size of local entries
  C->offdNNZ = C->offdRowStarts[N];   // update the size of non-local entries

  if (C->diagNNZ) C->diagCols = (dlong *) calloc(C->diagNNZ, sizeof(dlong));  // allocate memory for the local entries and set it to 0
  if (C->offdNNZ) C->offdCols = (dlong *) calloc(C->offdNNZ, sizeof(dlong));  // allocate memory for the non-local entries  and set it to 0

  // fill in the columns for strong connections
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    dfloat Aii = diagA[i];	     // compute abs value of A[i][i]

    dlong diagCounter = C->diagRowStarts[i]; // start of the local entries in "column i"
    dlong offdCounter = C->offdRowStarts[i]; // start of the non-local entries in "column i"

    //local entries
    C->diagCols[diagCounter++] = i;  // diag entry  // for each "node i"
    dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1]; // start & end of rows for "node i"
    for(dlong jj = Jstart+1; jj<Jend; jj++){     // loop over all entries in (start/end)
      dlong col = A->diagCols[jj];               
      dfloat Ajj = diagA[col];
      if( fabs(Ajj) > threshold*sqrt(fabs(Ajj*Aii)))      // compare the measure of each entry  wrt ( threshold * max )
        C->diagCols[diagCounter++] = A->diagCols[jj];  //  store the corresponding columns for  indices
    }
    // non-local entries   // for each "node i"
    Jstart = A->offdRowStarts[i], Jend = A->offdRowStarts[i+1];   // start & end of rows for "node i"
    for(dlong jj = Jstart; jj<Jend; jj++){    // loop over all entries in (start/end)
      dlong col = A->offdCols[jj];
      dfloat Ajj = diagA[col];      
      if(fabs(Ajj) > threshold*sqrt(fabs(Aii*Ajj)))      // compare the measure of each entry  wrt ( threshold * max )
        C->offdCols[offdCounter++] = A->offdCols[jj];  //  store the corresponding columns for  indices
    }
  }
  if(N) free(maxOD);

  // this exploit the format CSR, since C is fake matrix since it only create the indices corresponding to the strong connections
  // but not the entry which is 1 and usesless. The strong connections are: 

   /*for (dlong i = 0 ; i < N : i++){
 	for (dlong jj = C-diagRowStarts[i]+1; jj < C->diagRowStarts[i+1] ; jj++)
	    printf("\n(%d,%d)\n", i , C->diagCols[jj] );
	for (dlong jj = C->offdRowStarts[i] ; jj < C->offdRowStarts[i+1] ; jj++)
	    printf("\n(%d,%d)\n", i , C->offdCols[jj] );
     }*/		  
  
  printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("\n+++\t diagNNZ = %d \t  offdNNZ = %d \t   ++++++\n",C->diagNNZ,C->offdNNZ);
  printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   	
 
  return C;
}

hlong * form_aggregates2(agmgLevel *level, csr *C){

  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diagNNZ;
  const dlong offdNNZ = C->offdNNZ;
  
  printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("++ \t  N = %d   \t  M=%d   \t  (%d ,%d) \t ++ \n",N,M,diagNNZ,offdNNZ);
  printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  hlong *FineToCoarse = (hlong *) calloc(M, sizeof(hlong));
  for (dlong i =0;i<M;i++) FineToCoarse[i] = -1;

  int   *states = (int *)   calloc(M, sizeof(int));

  csr *A = level->A;
  hlong *globalRowStarts = level->globalRowStarts;

  int    *intSendBuffer;
  hlong  *hlongSendBuffer;
  dfloat *dfloatSendBuffer;
  if (level->A->NsendTotal) {
    intSendBuffer = (int *) calloc(A->NsendTotal,sizeof(int));
    hlongSendBuffer = (hlong *) calloc(A->NsendTotal,sizeof(hlong));
    dfloatSendBuffer = (dfloat *) calloc(A->NsendTotal,sizeof(dfloat));
  }


  for(dlong i=0; i<N; i++)
    states[i] = -1;

  // add the number of non-zeros in each column
  //local non-zeros

  int *nnzCnt, *recvNnzCnt;
  if (A->NHalo) nnzCnt = (int *) calloc(A->NHalo,sizeof(int));
  if (A->NsendTotal) recvNnzCnt = (int *) calloc(A->NsendTotal,sizeof(int));
  
  
  	dlong **diagN = new dlong*[N];        
	dlong diagN_index[N];
	
	dlong **offdN = new dlong*[N];        
	dlong offdN_index[N];
	
	dlong V[N];
	
  
  //construct local-strong neigboors
  for(dlong i=0; i<N; i++){  // should be Nlocal ?
	int NN = C->diagRowStarts[i+1] - C->diagRowStarts[i];  
	diagN[i] = (dlong *) calloc(NN,sizeof(dlong));
	diagN_index[i] = NN;
	for (int j=0 ; j < NN ; j++  ){
		diagN[i][j] =  C->diagCols[C->diagRowStarts[i] + j];
	}
  }
  
  /*
  printf("\n local neighborhood \n");
  for(int i=0;i<N;i++){
		printf("\n (i = %d, # = %d)\t",i,diagN_index[i]);
		for( int j=0;j<diagN_index[i];j++)
				printf("%d ",diagN[i][j]);
  }
  */
  
  //construct non-strong neigboors
  for(dlong i=0; i<N; i++){  // should be Nlocal ?
	int NN = C->offdRowStarts[i+1] - C->offdRowStarts[i];  
	offdN[i] = (dlong *) calloc(NN,sizeof(dlong));
	offdN_index[i] = NN;
	for (int j=0 ; j < NN ; j++  ){
		offdN[i][j] =  C->diagCols[C->diagRowStarts[i] + j];
	}
  }
  
  /*
  printf("\n non local neighborhood \n");
  for(int i=0;i<N;i++){
		printf("\n (i = %d, # = %d)\t",i,offdN_index[i]);
		for( int j=0;j<offdN_index[i];j++)
				printf("%d ",offdN[i][j]);
  }*/

  int R_num = 0;
  
  for (int i=0;i<N;i++)
	if(diagN_index[i]>1)
		R_num++;

  int R_nodes[R_num];
  int k = 0;
  
  for (int i=0;i<N;i++){
	if(diagN_index[i]>1){
		R_nodes[k] = i;
		k++;
	}
  }


 hlong done =0;
 int Agg_num = 0;
 
 /*
 printf("\n==============\n # agg=%d  N=%d \n=================\n",Agg_num,N);
 for(int i=0;i<N;i++)
	printf("\%d ",states[i]);
 */
	
	 #pragma omp parallel for
	 for(dlong i=0; i<R_num; i++){
		if (states[R_nodes[i]] == -1){
			int ok = 0;
			for(int j=0; j<diagN_index[R_nodes[i]];j++)
				ok = ok + states[diagN[R_nodes[i]][j]];
			if (ok == -diagN_index[R_nodes[i]]){
				for(int j=0; j<diagN_index[R_nodes[i]];j++)
					states[diagN[R_nodes[i]][j]] = Agg_num;				
				Agg_num++;
			}
		}	 
	 }
	
	/* 
printf("\n==============\n # agg=%d   N=%d \n=================\n",Agg_num,N);
 for(int i=0;i<N;i++)
	printf("\%d ",states[i]); 
*/

R_num=0;

for (int i=0;i<N;i++)
	if (states[i]==-1)
		R_num++;

k = 0;
for (int i=0;i<N;i++){
	if (states[i]==-1){
		R_nodes[k] = i;
		k++;
	}
} 

	 #pragma omp parallel for
	 for(dlong i=0; i<R_num; i++){
		if (states[R_nodes[i]] == -1){  // sanity check			
			if (diagN_index[R_nodes[i]]>1){
				int Agg_max;
				int posIndx = 0;
				int MoreAgg[Agg_num];
				for (int j=0;j<Agg_num;j++)
					MoreAgg[j]=0;
				for(int j=0; j<diagN_index[R_nodes[i]];j++)
					MoreAgg[states[diagN[R_nodes[i]][j]]]++;
					
				/*	
				printf("\n R-node = %d \n",R_nodes[i]);		
				for (int j=0;j<Agg_num;j++)
					printf("%d ",MoreAgg[j]);	
				printf("\n");	
				*/
					
				Agg_max = -1;
				for (int j=0;j<Agg_num;j++){
					if (Agg_max < MoreAgg[j]){
						Agg_max = MoreAgg[j];
						posIndx = j;
					}
				}
				//printf("\n   Agg_max = %d pos = %d \n",Agg_max,posIndx);		
				
				states[R_nodes[i]] = posIndx;				
			}
			else{  // nodo aislado
				states[R_nodes[i]] = Agg_num;
				Agg_num++;								
			}			
		}	 
	 }

/*
printf("\n==============\n # agg=%d   N=%d \n=================\n",Agg_num,N);
 for(int i=0;i<N;i++)
	printf("\%d ",states[i]); 
*/


/*
  hlong done = 0;
  while(!done){
    // first neighbours
    		
		#pragma omp parallel for
		for(dlong i=0; i<N; i++){
			//local entries
			for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
			  const dlong col = C->diagCols[jj];
			  if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
				smax = states[col];
				rmax = rands[col];
				imax = col + globalRowStarts[rank];
			  }
			}
			//nonlocal entries
			for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
			  const dlong col = C->offdCols[jj];
			  if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])) {
				smax = states[col];
				rmax = rands[col];
				imax = A->colMap[col];
			  }
			}
		  }
		  Ts[i] = smax;
		  Tr[i] = rmax;
		  Ti[i] = imax;
		}
	

    //share results
    csrHaloExchange(A, sizeof(dfloat), Tr, dfloatSendBuffer, Tr+A->NlocalCols);
    csrHaloExchange(A, sizeof(int), Ts, intSendBuffer, Ts+A->NlocalCols);
    csrHaloExchange(A, sizeof(hlong), Ti, hlongSendBuffer, Ti+A->NlocalCols);

    // second neighbours
    #pragma omp parallel for
    for(dlong i=0; i<N; i++){
      int    smax = Ts[i];
      dfloat rmax = Tr[i];
      hlong  imax = Ti[i];

      //local entries
      for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
        const dlong col = C->diagCols[jj];
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
        const dlong col = C->offdCols[jj];
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }

      // if I am the strongest among all the 1 and 2 ring neighbours
      // I am an MIS node
      if((states[i] == 0) && (imax == (i + globalRowStarts[rank])))
        states[i] = 1;
      
      // if there is an MIS node within distance 2, I am removed
      if((states[i] == 0) && (smax == 1))
        states[i] = -1;
    }

/*/

    csrHaloExchange(A, sizeof(int), states, intSendBuffer, states+A->NlocalCols);

  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));
  level->globalAggStarts = (hlong *) calloc(size+1,sizeof(hlong));
  // count the coarse nodes/aggregates
  
    MPI_Allgather(&Agg_num,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,agmg::comm);

  level->globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    level->globalAggStarts[r+1] = level->globalAggStarts[r] + gNumAggs[r];

  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    FineToCoarse[i] = level->globalAggStarts[rank] + states[i];

  //share the initial aggregate flags
  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);

 //////////////////////////////////////////////////////////////////////////////////////////////
 // print FineToCoarse
 /////////////////////////////////////////////////////////////////////////////////////////////
/* printf("\n FineToCoarse...  numAgg = %d \n",Agg_num);
  for(dlong i=0; i<N; i++)
	printf("%d \t ",FineToCoarse[i]);
  printf("\n-------------------\n");
*/


/*
  // form the aggregates
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int   smax = states[i];
    dfloat rmax = rands[i];
    hlong  imax = i + globalRowStarts[rank];
    hlong  cmax = FineToCoarse[i];

    if(smax != 1){
      //local entries
      for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
        const dlong col = C->diagCols[jj];
        if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
          smax = states[col];
          rmax = rands[col];
          imax = col + globalRowStarts[rank];
          cmax = FineToCoarse[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
        const dlong col = C->offdCols[jj];
        if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])){
          smax = states[col];
          rmax = rands[col];
          imax = A->colMap[col];
          cmax = FineToCoarse[col];
        }
      }
    }
    Ts[i] = smax;
    Tr[i] = rmax;  
    Ti[i] = imax;
    Tc[i] = cmax;

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }


printf("\n-------------------\n");
 printf("\n FineTo Coarse 1st comparison  %d\n",rank);
  for(dlong i=0; i<N; i++)
	printf("%d \t ",FineToCoarse[i]);
  printf("\n-------------------\n");

  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);
  csrHaloExchange(A, sizeof(dfloat), Tr, dfloatSendBuffer, Tr+A->NlocalCols);
  csrHaloExchange(A, sizeof(int), Ts, intSendBuffer, Ts+A->NlocalCols);
  csrHaloExchange(A, sizeof(hlong), Ti, hlongSendBuffer, Ti+A->NlocalCols);
  csrHaloExchange(A, sizeof(hlong), Tc, hlongSendBuffer, Tc+A->NlocalCols);

  // second neighbours
  #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int    smax = Ts[i];
    dfloat rmax = Tr[i];
    hlong  imax = Ti[i];
    hlong  cmax = Tc[i];

    //local entries
    for(dlong jj=C->diagRowStarts[i]+1;jj<C->diagRowStarts[i+1];jj++){
      const dlong col = C->diagCols[jj];
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }
    //nonlocal entries
    for(dlong jj=C->offdRowStarts[i];jj<C->offdRowStarts[i+1];jj++){
      const dlong col = C->offdCols[jj];
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }

printf("\n-------------------\n");
 printf("\n FineToCoarse 2nd comparison...  %d\n",rank);
  for(dlong i=0; i<N; i++)
	printf("%d \t ",FineToCoarse[i]);
  printf("\n-------------------\n");



  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);

  free(rands);
  free(states);
  free(Tr);
  free(Ts);
  free(Ti);
  free(Tc);   */
  if (level->A->NsendTotal) {
    free(intSendBuffer);
    free(hlongSendBuffer);
    free(dfloatSendBuffer);
  }

  //TODO maybe free C here?
// print aggregates info
/* printf("\n++++++++++++++\n  N agg = %d  \n+++++++++++++++++++\n",Agg_num);
  int N_agg[Agg_num];
  for (int i=0;i<Agg_num;i++){
	  N_agg[i]=0;
  }for (int i=0;i<N;i++){
	  N_agg[FineToCoarse[i]]++;
  }for (int i=0;i<Agg_num;i++){
	  printf("\n Agg = %d  => %d",i,N_agg[i]);
  }
  
  */


  return FineToCoarse;
  
}


csr *construct_interpolator2(agmgLevel *level, hlong *FineToCoarse, dfloat **nullCoarseA){
  // MPI info
  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  const dlong N = level->A->Nrows;
  // const dlong M = level->A->Ncols;

  hlong *globalAggStarts = level->globalAggStarts;

  const hlong globalAggOffset = level->globalAggStarts[rank];
  const dlong NCoarse = (dlong) (globalAggStarts[rank+1]-globalAggStarts[rank]); //local num agg

  csr* P = (csr *) calloc(1, sizeof(csr));

  P->Nrows = N;
  P->Ncols = NCoarse;

  P->NlocalCols = NCoarse;
  P->NHalo = 0;

  P->diagRowStarts = (dlong *) calloc(N+1, sizeof(dlong));
  P->offdRowStarts = (dlong *) calloc(N+1, sizeof(dlong));

  // each row has exactly one nonzero per row
  P->diagNNZ =0;
  P->offdNNZ =0;
  for(dlong i=0; i<N; i++) {
    hlong col = FineToCoarse[i];
    if ((col>globalAggOffset-1)&&(col<globalAggOffset+NCoarse)) {
      P->diagNNZ++;
      P->diagRowStarts[i+1]++;
    } else {
      P->offdNNZ++;
      P->offdRowStarts[i+1]++;
    }
  }
  for(dlong i=0; i<N; i++) {
    P->diagRowStarts[i+1] += P->diagRowStarts[i];
    P->offdRowStarts[i+1] += P->offdRowStarts[i];
  }

  if (P->diagNNZ) {
    P->diagCols  = (dlong *)  calloc(P->diagNNZ, sizeof(dlong));
    P->diagCoefs = (dfloat *) calloc(P->diagNNZ, sizeof(dfloat));
  }
  
  hlong *offdCols;
  if (P->offdNNZ) {
    offdCols  = (hlong *)  calloc(P->offdNNZ, sizeof(hlong));
    P->offdCols  = (dlong *)  calloc(P->offdNNZ, sizeof(dlong));
    P->offdCoefs = (dfloat *) calloc(P->offdNNZ, sizeof(dfloat));
  }

  dlong diagCnt = 0;
  dlong offdCnt = 0;
  for(dlong i=0; i<N; i++) {
    hlong col = FineToCoarse[i];
    if ((col>globalAggStarts[rank]-1)&&(col<globalAggStarts[rank+1])) {
      P->diagCols[diagCnt] = (dlong) (col - globalAggOffset); //local index
      P->diagCoefs[diagCnt++] = 1;  //level->A->null[i];
    }else{
      offdCols[offdCnt] = col;
      P->offdCoefs[offdCnt++] = 1;  //level->A->null[i];
    }
  }

  //record global indexing of columns
  P->colMap = (hlong *)   calloc(P->Ncols, sizeof(hlong));
  for (dlong i=0;i<P->Ncols;i++)
    P->colMap[i] = i + globalAggOffset;

  if (P->offdNNZ) {
    //we now need to reorder the x vector for the halo, and shift the column indices
    hlong *col = (hlong *) calloc(P->offdNNZ,sizeof(hlong));
    for (dlong i=0;i<P->offdNNZ;i++)
      col[i] = offdCols[i]; //copy non-local column global ids

    //sort by global index
    std::sort(col,col+P->offdNNZ);

    //count unique non-local column ids
    P->NHalo = 0;
    for (dlong i=1;i<P->offdNNZ;i++)
      if (col[i]!=col[i-1])  col[++P->NHalo] = col[i];
    P->NHalo++; //number of unique columns

    P->Ncols += P->NHalo;

    //save global column ids in colMap
    P->colMap = (hlong *) realloc(P->colMap, P->Ncols*sizeof(hlong));
    for (dlong i=0; i<P->NHalo; i++)
      P->colMap[i+P->NlocalCols] = col[i];
    free(col);

    //shift the column indices to local indexing
    for (dlong i=0;i<P->offdNNZ;i++) {
      hlong gcol = offdCols[i];
      for (dlong m=P->NlocalCols;m<P->Ncols;m++) {
        if (gcol == P->colMap[m])
          P->offdCols[i] = m;
      }
    }
    free(offdCols);
  }

  csrHaloSetup(P,globalAggStarts);


 // normalize the columns of P
  *nullCoarseA = (dfloat *) calloc(P->Ncols,sizeof(dfloat));

  //add local nonzeros
  for(dlong i=0; i<P->diagNNZ; i++)
    (*nullCoarseA)[P->diagCols[i]] += P->diagCoefs[i] * P->diagCoefs[i];

  dfloat *nnzSum, *recvNnzSum;
  if (P->NHalo) nnzSum = (dfloat *) calloc(P->NHalo,sizeof(dfloat));
  if (P->NsendTotal) recvNnzSum = (dfloat *) calloc(P->NsendTotal,sizeof(dfloat));

  //add the non-local non-zeros
  for (dlong i=0;i<P->offdNNZ;i++)
    nnzSum[P->offdCols[i]-P->NlocalCols] += P->offdCoefs[i] * P->offdCoefs[i];

  //do a reverse halo exchange
  int tag = 999;

  // initiate immediate send  and receives to each other process as needed
  dlong recvOffset = 0;
  dlong sendOffset = 0;
  int sendMessage = 0, recvMessage = 0;
  for(int r=0;r<size;++r){
    if (P->NsendTotal) {
      if(P->NsendPairs[r]) {
        MPI_Irecv(recvNnzSum+sendOffset, P->NsendPairs[r], MPI_DFLOAT, r, tag,
            agmg::comm, (MPI_Request*)P->haloSendRequests+sendMessage);
        sendOffset += P->NsendPairs[r];
        ++sendMessage;
      }
    }
    if (P->NrecvTotal) {
      if(P->NrecvPairs[r]){
        MPI_Isend(nnzSum+recvOffset, P->NrecvPairs[r], MPI_DFLOAT, r, tag,
            agmg::comm, (MPI_Request*)P->haloRecvRequests+recvMessage);
        recvOffset += P->NrecvPairs[r];
        ++recvMessage;
      }
    }
  }

  // Wait for all sent messages to have left and received messages to have arrived
  if (P->NrecvTotal) {
    MPI_Status *sendStatus = (MPI_Status*) calloc(P->NsendMessages, sizeof(MPI_Status));
    MPI_Waitall(P->NsendMessages, (MPI_Request*)P->haloSendRequests, sendStatus);
    free(sendStatus);
  }
  if (P->NsendTotal) {
    MPI_Status *recvStatus = (MPI_Status*) calloc(P->NrecvMessages, sizeof(MPI_Status));
    MPI_Waitall(P->NrecvMessages, (MPI_Request*)P->haloRecvRequests, recvStatus);
    free(recvStatus);
  }

  for(dlong i=0;i<P->NsendTotal;++i){
    // local index of outgoing element in halo exchange
    dlong id = P->haloElementList[i];
    (*nullCoarseA)[id] += recvNnzSum[i];
  }

  if (P->NHalo) free(nnzSum);

  for(dlong i=0; i<NCoarse; i++)
    (*nullCoarseA)[i] = 1; //sqrt((*nullCoarseA)[i]);

  csrHaloExchange(P, sizeof(dfloat), *nullCoarseA, P->sendBuffer, *nullCoarseA+P->NlocalCols);

  for(dlong i=0; i<P->diagNNZ; i++)
    P->diagCoefs[i] =1;   // /= (*nullCoarseA)[P->diagCols[i]];
  for(dlong i=0; i<P->offdNNZ; i++)
    P->offdCoefs[i] =1; // /= (*nullCoarseA)[P->offdCols[i]];

  MPI_Barrier(agmg::comm);
  if (P->NsendTotal) free(recvNnzSum);

  return P;
}



csr *OneSmooth(agmgLevel *level, double w,csr *A,csr *P){
		// P is already transposed
		int rank,size;
		rank = agmg::rank;
		size = agmg::size;
		
		printf("\n+++++++++++++++++\n");
		printf("\n+++ OneSmooth +++\n");
		printf("\n+++++++++++++++++\n");
				
		printf("\n+++++++++++++++++\n Recived info A =  %d x %d , P = %d x %d ",A->Nrows,A->Ncols,P->Nrows,P->Ncols);
		
		
	/*	printf("\n+++++++++++++++++\n print P' indices\n ++++++++++++++++++++++++++\n ");
			
       for(dlong i=0; i<P->Nrows; i++){
			dlong Jstart = P->diagRowStarts[i], Jend = P->diagRowStarts[i+1]; // start & end of rows for "node i"
			for(dlong jj = Jstart; jj<Jend; jj++){
				printf("(%d,%d)  ",i,P->diagCols[jj]);
			}
			printf("\t\t[row %d] \n",i);
		}
		printf("\n+++++++++++++++++\n ");	
	*/
		
		
		// costruct F  (final product)
		csr* F = (csr *) calloc(1, sizeof(csr));
		F->Nrows = A->Nrows;
		F->Ncols = P->Nrows;    // P is transposed !
		
		printf("\n+++++++++++++++++\n F constructed %d x %d \n",F->Nrows,F->Ncols);
		
		// construct R = I - w*D^{-1}*A
		csr* Re = (csr *) calloc(1, sizeof(csr));
		Re->Nrows = A->Nrows;
		Re->Ncols = A->Ncols;
		Re->diagNNZ = A->diagNNZ;
		printf("\n+++++++++++++++++\n Re constructed %d x %d \n ++++++++++ \n",Re->Nrows,Re->Ncols);
		
		dlong N = Re->Nrows;      // A symetric 
		dlong M = F->Ncols;       // P columns
		dlong NNZ = Re->diagNNZ;
			
		F->diagRowStarts = (dlong *) calloc(N+1, sizeof(dlong));			
		F->diagRowStarts[0]=0;
		Re->diagRowStarts = (dlong *) calloc(N+1, sizeof(dlong));
		Re->diagCols = (dlong *) calloc(NNZ, sizeof(dlong));
		Re->diagCoefs = (dfloat *) calloc(NNZ, sizeof(dfloat));
		
	//	printf("\nRowStarts +++++++++++++++++\n ");
		for (int i=0;i<N+1;i++){
			Re->diagRowStarts[i] = A->diagRowStarts[i];		
			//printf("\%d  ",Re->diagRowStarts[i]);
		}
	//	printf("\n+++++++++++++++++\n ");
		
	//	printf("\nCols +++++++++++++++++\n ");
		for (int i=0;i<NNZ;i++){
			Re->diagCols[i] = A->diagCols[i];		
			//printf("\%d  ",Re->diagCols[i]);
		}
	//	printf("\n+++++++++++++++++\n ");
		
		
		
	//	printf("\n diag(A) +++++++++++++++++\n ");
		//store the diagonal of A for all needed columns
		dfloat *diagA = (dfloat *) calloc(N,sizeof(dfloat));
		for (dlong i=0;i<N;i++){
			diagA[i] = A->diagCoefs[A->diagRowStarts[i]];
		//	printf("\%.2f  ",diagA[i]);
		}
		//printf("\n+++++++++++++++++\n ");



		// multiply by -w*D^{-1}
		for(dlong i=0; i<N; i++){
			dlong Jstart = A->diagRowStarts[i], Jend = A->diagRowStarts[i+1]; // start & end of rows for "node i"
			for(dlong jj = Jstart; jj<Jend; jj++){
				Re->diagCoefs[jj] = -(w/diagA[i])*A->diagCoefs[jj];
			//	printf("%.2f ",Re->diagCoefs[jj]);
			}
			//printf("\t(Fila %d) \n",i);
		}
		//printf("\n+++++++++++++++++\n ");
		
		
		// add Identity 
		for (dlong i=0;i<N;i++){
			//printf("\n%.3f \t",Re->diagCoefs[Re->diagRowStarts[i]]);
			Re->diagCoefs[Re->diagRowStarts[i]] =  Re->diagCoefs[Re->diagRowStarts[i]] + 1;
			//printf(" %.3f \n",Re->diagCoefs[Re->diagRowStarts[i]]);
		}	
		/*
		printf("\n  IMp con NNZ = %d +++++++++++++++++\n ",NNZ);	
		int s=0;
		for (dlong i=0;i<NNZ;i++){
			if (i < Re->diagRowStarts[s+1])
				printf("%.2f ",Re->diagCoefs[i]);
			
			if (i == Re->diagRowStarts[s+1]){
				printf("\n%.2f ",Re->diagCoefs[i]);
				s++;
			}
		}*/
		////////////////////////////////////////////////////////////////////////////////////////		
		printf("\n+++++++++++++++++\n Sanity check  indices  => N = %d     M=%d\n ",N,M);			
		printf("\n-------------Sanity check on P \n NNZ = %d \n-----------------\n",P->diagNNZ);
		/*for (int k =0; k<M+1;k++)
			printf("%d ",P->diagRowStarts[k]);
		printf("\n------------------------------\n");
		*/
				
		// multiply R by P
		int NNZeros = 0;
		for(dlong i=0; i<N; i++){  // filas in Re 
			dlong Istart = Re->diagRowStarts[i], Iend = Re->diagRowStarts[i+1]; // row i" in R
			int countE = 0;
			int N1 = Iend - Istart; 
					
			/*printf("\n Fila %d original  \n",i);
			for (int k=0;k<N1;k++){
				printf("%d  ",Re->diagCols[Istart+k]);
			} */ 
			
			int ColI[N1];						
			int insert = 0;
			for (int k=0;k<N1;k++){
				if (i > 0 && i < N-1){
					if ( Re->diagCols[Istart+k+1] < i )
						ColI[k] = Istart+k+1;	
					else if (insert == 0){
						ColI[k] = Istart;
						insert = 1;
					}
					else
						ColI[k] = Istart+k;	
				}else if (i == 0 )
						ColI[k] = Istart+k;
				else{
					if (k < N1-1)
						ColI[k] = Istart+k+1;		
					else
						ColI[k] = Istart;		
				}
			}
			
		 /* printf("\n Fila i = %d ordenada  \n",i);
			for (int k=0;k<N1;k++){
				printf("%d  ",Re->diagCols[ColI[k]]);
		  }*/				
			int NZrows=0;			
			for(dlong j =0; j < M; j++){ // filas en P'					
				dlong Jstart = P->diagRowStarts[j], Jend = P->diagRowStarts[j+1]; // column j in P						
				int N2 = Jend - Jstart;						
			/*	printf("\n Fila j = %d\n",j);
				for (int k=0;k<N2;k++){
					printf("%d  ",P->diagCols[Jstart+k]);
				}
				printf("\n"); */
				int posI = 0,posJ=0;
				dfloat Sum =0;
				int Ival = Re->diagCols[ColI[0]];
				int Jval = P->diagCols[Jstart];
				while ( posI<N1 && posJ<N2 ){					
					//printf("\n posI=%d  valI=%d  posJ=%d  valJ=%d",posI,Ival,posJ,Jval);
					
						if (Ival < Jval){
							posI++;
							if(posI<N1) Ival = Re->diagCols[ColI[posI]];
						}
						else if (Ival > Jval){
							posJ++;
							if(posJ<N2) Jval = P->diagCols[Jstart+posJ];
						}
						else{
							//printf("\n col j=%d  [ %d (%d) = %d (%d) ]",j,Ival,posI,Jval,posJ);
							Sum = Sum + Re->diagCoefs[ColI[posI]]*P->diagCoefs[Jstart+posJ];
							posI++;
							posJ++;
							if(posI<N1) Ival = Re->diagCols[ColI[posI]];
							if(posJ<N2) Jval = P->diagCols[Jstart+posJ];
						}	
														
				}
				//printf("\n Sum = %.3f\n",Sum);
				if (Sum != 0){ NZrows++; NNZeros++;}
				//printf("\n============\n Nzero =%d \t Nrows = %d\n============\n",Nzero,Nrows);
			}
	//		printf("\n NZrows = %d",NZrows);
			F->diagRowStarts[i+1] = F->diagRowStarts[i] + NZrows;
		}
		
		printf("\n+++++++++++\n");
		for (int k =0; k<N+1;k++)
			printf("%d ",F->diagRowStarts[k]);
		printf("\n+++++++++++ check NNZeros = %d \n",NNZeros);
	
		F->diagCoefs = (dfloat *) calloc(NNZeros, sizeof(dfloat));  
		F->diagCols = (dlong *) calloc(NNZeros, sizeof(dlong));  
		F->diagNNZ = NNZeros;
		
		dlong pos = 0;
	
		for(dlong i=0; i<N; i++){  // filas in Re 
			dlong Istart = Re->diagRowStarts[i], Iend = Re->diagRowStarts[i+1]; // row i" in R
			int countE = 0;
			int N1 = Iend - Istart; 
							
			int ColI[N1];						
			int insert = 0;
			for (int k=0;k<N1;k++){
				if (i > 0 && i < N-1){
					if ( Re->diagCols[Istart+k+1] < i )
						ColI[k] = Istart+k+1;	
					else if (insert == 0){
						ColI[k] = Istart;
						insert = 1;
					}
					else
						ColI[k] = Istart+k;	
				}else if (i == 0 )
						ColI[k] = Istart+k;
				else{
					if (k < N1-1)
						ColI[k] = Istart+k+1;		
					else
						ColI[k] = Istart;		
				}
			}
			
			printf("\n Fila i = %d  \n",i);   
							
			//int NZrows=0;			
			for(dlong j =0; j < M; j++){ // filas en P'					
				dlong Jstart = P->diagRowStarts[j], Jend = P->diagRowStarts[j+1]; // column j in P						
				int N2 = Jend - Jstart;						
				int posI = 0,posJ=0;
				dfloat Sum =0;
				int Ival = Re->diagCols[ColI[0]];
				int Jval = P->diagCols[Jstart];
				while ( posI<N1 && posJ<N2 ){					
						if (Ival < Jval){
							posI++;
							if(posI<N1) Ival = Re->diagCols[ColI[posI]];
						}
						else if (Ival > Jval){
							posJ++;
							if(posJ<N2) Jval = P->diagCols[Jstart+posJ];
						}
						else{
							//printf("\n col j=%d  [ %d (%d) = %d (%d) ]",j,Ival,posI,Jval,posJ);
							Sum = Sum + Re->diagCoefs[ColI[posI]]*P->diagCoefs[Jstart+posJ];
							posI++;
							posJ++;
							if(posI<N1) Ival = Re->diagCols[ColI[posI]];
							if(posJ<N2) Jval = P->diagCols[Jstart+posJ];
						}	
														
				}
				if (Sum != 0){ 
					printf("\n (I-wD^(-1)A)(%d,%d) Sum = %.3f  pos = %d ",i,j,Sum,pos);
					F->diagCoefs[pos] = Sum;
					F->diagCols[pos] = j;
					pos++;
				}				
			}
			//printf("\n NZrows = %d",NZrows);
			//F->diagRowStarts[i+1] = F->diagRowStarts[i] + NZrows;
		}		
	
	
		
		printf("\n Sanity check pos = %d = %d = NNZeros \n",pos,NNZeros);
	
		free(Re);
		
		F->NlocalCols = F->Ncols; 
		F->NHalo = 0;
		F->offdNNZ = 0;
		
		F->offdRowStarts = (dlong *) calloc(N+1, sizeof(dlong));	// zeros not elements not yet
		F->offdCoefs = (dfloat *) calloc(NNZeros, sizeof(dfloat));  
		F->offdCols = (dlong *) calloc(NNZeros, sizeof(dlong));  
		F->offdNNZ = 0;
		
			
		//////////////////////////////
		printf("\n\n Sanity check \n\n");
		for (dlong i=0;i<F->Nrows;i++)
			printf("\n (i=%d)  %d to %d ... ",i, F->diagRowStarts[i],F->diagRowStarts[i+1]);
	
		for (dlong i=0;i<F->diagNNZ;i++)
			printf("\n %d %d ",i, F->diagCols[i]);
	
	
	
		return F;
		
}



typedef struct{
	dlong index;
	dlong Nnbs;
	dlong *nbs;
} nbs_t;


int compareNBS(const void *a, const void *b){
	nbs_t *pa = (nbs_t *)a;	
	nbs_t *pb = (nbs_t *)b;
	
	
	if (pa->Nnbs < pb->Nnbs)	return +1;
	if (pa->Nnbs > pb->Nnbs)	return -1;
	if (pa->index < pa->index )	return +1;
	if (pa->index > pa->index )	return -1;
	
	return 0;
}



hlong * form_aggregates3(agmgLevel *level, csr *C,setupAide options){

  int rank, size;
  rank = agmg::rank;
  size = agmg::size;

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diagNNZ;
  const dlong offdNNZ = C->offdNNZ;
  
  printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++\n");
  printf("++ \t  N = %d   \t  M=%d   \t  (%d ,%d) \t ++ \n",N,M,diagNNZ,offdNNZ);
  printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  hlong *FineToCoarse = (hlong *) calloc(M, sizeof(hlong));
  for (dlong i =0;i<M;i++) FineToCoarse[i] = -1;

  int   *states = (int *)   calloc(M, sizeof(int));

  csr *A = level->A;
  hlong *globalRowStarts = level->globalRowStarts;

  int    *intSendBuffer;
  hlong  *hlongSendBuffer;
  dfloat *dfloatSendBuffer;
  if (level->A->NsendTotal) {
    intSendBuffer = (int *) calloc(A->NsendTotal,sizeof(int));
    hlongSendBuffer = (hlong *) calloc(A->NsendTotal,sizeof(hlong));
    dfloatSendBuffer = (dfloat *) calloc(A->NsendTotal,sizeof(dfloat));
  }


  for(dlong i=0; i<N; i++)
    states[i] = -1;

  // add the number of non-zeros in each column
  //local non-zeros

  int *nnzCnt, *recvNnzCnt;
  if (A->NHalo) nnzCnt = (int *) calloc(A->NHalo,sizeof(int));
  if (A->NsendTotal) recvNnzCnt = (int *) calloc(A->NsendTotal,sizeof(int));
  
  
  nbs_t *V = (nbs_t *) calloc(N,sizeof(nbs_t));
	
	 
  //construct local-strong neigboors
  for(dlong i=0; i<N; i++){  // should be Nlocal ?
	V[i].index = i;
	V[i].Nnbs  = C->diagRowStarts[i+1] - C->diagRowStarts[i];  
	V[i].nbs   = (dlong *) calloc(V[i].Nnbs,sizeof(dlong));
	for (int j=0 ; j < V[i].Nnbs ; j++  ){
		V[i].nbs[j] =  C->diagCols[C->diagRowStarts[i] + j];
	}
  }
  
  /*
  printf("\n local neighborhood befor qsort \n");
  for(int i=0;i<N;i++){
		printf("\n (i = %d, # = %d)\t",V[i].index,V[i].Nnbs);
		for( int j=0;j<V[i].Nnbs;j++)
				printf("%d ",V[i].nbs[j]);
  } 
  */


// sort V base on something  
int MySort = 0;

options.getArgs("SORT",MySort);

if (MySort>0)	qsort(V,N,sizeof(nbs_t),compareNBS);
	
/*	
  printf("\n local neighborhood after qsort \n");
  for(int i=0;i<N;i++){
		printf("\n (i = %d, # = %d)\t",V[i].index,V[i].Nnbs);
		for( int j=0;j<V[i].Nnbs;j++)
				printf("%d ",V[i].nbs[j]);
  }
  */
   
  int R_num = 0;
  
   
  for (int i=0;i<N;i++)
	if(V[i].Nnbs>1)
		R_num++;


	

  int R_nodes[R_num];
  int R_pos[R_num];
  int k = 0;
  
  for (int i=0;i<N;i++){
	if(V[i].Nnbs>1){
		R_nodes[k] = V[i].index;
		R_pos[k] = i;
		k++;
	}
  }


 hlong done =0;
 int Agg_num = 0;
 
 /*
 printf("\n==============\n # agg=%d  N=%d \n=================\n",Agg_num,N);
 for(int i=0;i<N;i++)
	printf("\%d ",states[i]);
 */
	
	 #pragma omp parallel for
	 for(dlong i=0; i<R_num; i++){
		if (states[R_nodes[i]] == -1){
			int ok = 0;
			//printf("\n  node %d free checking nbs \n",R_nodes[i]);
			for(int j=1; j<V[R_pos[i]].Nnbs;j++){ // 0 is itself
				//printf("%d ",V[R_pos[i]].nbs[j]);
				if (states[V[R_pos[i]].nbs[j]]>-1){
					ok=1;
					j = V[R_nodes[i]].Nnbs +10;
				}
			}
				//ok = ok + states[V[R_nodes[i]].nbs[j]];
			//if (ok == -V[R_nodes[i]].Nnbs){
			if (ok == 0){
				//printf("\n neighborhood %d   libre....\n",R_nodes[i]);
				for(int j=0; j<V[R_pos[i]].Nnbs;j++){
					states[V[R_pos[i]].nbs[j]] = Agg_num;				
				}
				
				//for(int i=0;i<N;i++)
				//	printf("%d(%d) ",i,states[i]); 
				
				Agg_num++;
			}
		}	 
	 }
	 
	 printf("\n\n>>>>>>  N 1st aggregates = %d >>>>>>>>\n\n",Agg_num);
	
/*	 
printf("\n==============\n # agg=%d   N=%d \n=================\n",Agg_num,N);
 for(int i=0;i<N;i++)
	printf("nodo = %d   agg =%d\n",i,states[i]); 
*/

R_num=0;

for (int i=0;i<N;i++)   // number of non-aggregate nodes
	if (states[i]==-1)
		R_num++;

k = 0;
for (int i=0;i<N;i++){  // update list of  non-agreggate nodes
	if (states[V[i].index]==-1){
		R_nodes[k] = V[i].index;
		R_pos[k] = i;
		k++;
	}
} 

int *psudoAgg = (int *) calloc(N,sizeof(int));

for (dlong i=0;i<N;i++)
	if (states[V[i].index]>-1)
		psudoAgg[states[V[i].index]]++;
	
/*
printf("\n------------------------------------------\n");
for(dlong i=0; i<Agg_num; i++)
	printf("\n Aggregate = %d, Naggr =%d ",i,psudoAgg[i]);
printf("\n------------------------------------------\n");
*/
	 #pragma omp parallel for
	 for(dlong i=0; i<R_num; i++){
		if (states[R_nodes[i]] == -1){  // sanity check			
			if (V[R_pos[i]].Nnbs>1){  // at most one neigbor
				int Agg_max;
				int posIndx = 0;
				int MoreAgg[Agg_num];
				for (int j=0;j<Agg_num;j++)
					MoreAgg[j]=0;
					
				//printf("\n checking nbs.... \n");	
				for(int j=1; j<V[R_pos[i]].Nnbs;j++){  // index 0 is itself
					if (states[V[R_pos[i]].nbs[j]] > -1){
						MoreAgg[states[V[R_pos[i]].nbs[j]]]++;
						//printf("%d ",V[R_pos[i]].nbs[j]);	
					}
				}
					
				/*	
				printf("\n R-node = %d \n",R_nodes[i]);		
				for (int j=0;j<Agg_num;j++)
					printf("%d ",MoreAgg[j]);	
				printf("\n");	
				*/
					
				Agg_max = -1;
				for (int j=0;j<Agg_num;j++){
					if (Agg_max <= MoreAgg[j]){
						if (j == 0){
							Agg_max = MoreAgg[j];
							posIndx = j;
						}
						else if (Agg_max < MoreAgg[j]){							
							Agg_max = MoreAgg[j];
							posIndx = j;
						}
						else if (psudoAgg[posIndx] > psudoAgg[j]){
							Agg_max = MoreAgg[j];
							posIndx = j;
						}
					}
				}
				//printf("\n  Agg_max =%d en posIndx = %d \n",Agg_max,posIndx);		
				states[R_nodes[i]] = posIndx;				
				//printf("\n  node %d added to agg =%d \n",R_nodes[i],posIndx);		
				psudoAgg[posIndx]++;
			}
			else{  // nodo aislado
				states[R_nodes[i]] = Agg_num;
				psudoAgg[Agg_num]++;
				Agg_num++;			
									
			}			
		}	 
	 }
	 
	 printf("\n\n>>>>>>  N 2nd aggregates = %d >>>>>>>>\n\n",Agg_num);

   // csrHaloExchange(A, sizeof(int), states, intSendBuffer, states+A->NlocalCols); como es local no deberia haber esto

  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));
  level->globalAggStarts = (hlong *) calloc(size+1,sizeof(hlong));
  // count the coarse nodes/aggregates
  
    MPI_Allgather(&Agg_num,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,agmg::comm);

  level->globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    level->globalAggStarts[r+1] = level->globalAggStarts[r] + gNumAggs[r];

  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    FineToCoarse[i] = level->globalAggStarts[rank] + states[i];

  //share the initial aggregate flags
  csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);

 //////////////////////////////////////////////////////////////////////////////////////////////
 // print FineToCoarse
 /////////////////////////////////////////////////////////////////////////////////////////////
/* printf("\n FineToCoarse...  numAgg = %d \n",Agg_num);
  for(dlong i=0; i<N; i++)
	printf("%d \t ",FineToCoarse[i]);
  printf("\n-------------------\n");
*/

  if (level->A->NsendTotal) {
    free(intSendBuffer);
    free(hlongSendBuffer);
    free(dfloatSendBuffer);
  }


/*	free(V);
	free(R_nodes);
	free(R_pos);
*/
  //TODO maybe free C here?
// print aggregates info

/*
 printf("\n++++++++++++++\n  N agg = %d  \n+++++++++++++++++++\n",Agg_num);
  int N_agg[Agg_num];
  for (int i=0;i<Agg_num;i++){
	  N_agg[i]=0;
  }for (int i=0;i<N;i++){
	  N_agg[FineToCoarse[i]]++;
  }for (int i=0;i<Agg_num;i++){
	  printf("\n Agg = %d  => %d",i,N_agg[i]);
  }  
  
  */


  return FineToCoarse;
  
}





