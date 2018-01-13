#include "boltzmann2D.h"

// complete a time step using LSERK4
void boltzmannSRABStep2D(mesh2D *mesh, iint tstep, iint haloBytes,
                        dfloat * sendBuffer, dfloat *recvBuffer, char * options){


 //  
  iint one = 1;
  iint zero = 0;
  if(tstep<3){
  
   dfloat *tmp;  occa::memory o_rhsqtmp; 
   tmp       = (dfloat*) calloc(mesh->Nelements*mesh->Np*mesh->Nfields, sizeof(dfloat));
   o_rhsqtmp = mesh->device.malloc(mesh->Np*mesh->Nelements*mesh->Nfields*sizeof(dfloat), tmp);
  
 
   // Compute RHS with LSERK q

 // intermediate stage time
  dfloat t = mesh->startTime + (tstep+1)*mesh->dt;

 // COMPUTE RAMP FUNCTION 
  dfloat ramp, drampdt;
  boltzmannRampFunction2D(t, &ramp, &drampdt);

 if (mesh->nonPmlNelements)  
      mesh->volumeKernel(mesh->nonPmlNelements,
                      mesh->o_nonPmlElementIds,
                      ramp, 
                      drampdt,
                      mesh->Nrhs,
                      mesh->shiftIndex,
                      mesh->o_vgeo,
                      mesh->o_DrT,
                      mesh->o_DsT,
                      mesh->o_q,
                      mesh->o_rhsq);

  if (mesh->pmlNelements)
     mesh->pmlVolumeKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          ramp, 
                          drampdt,
                          mesh->Nrhs,
                          mesh->shiftIndex,
                          mesh->o_vgeo,
                          mesh->o_pmlSigmaX,
                          mesh->o_pmlSigmaY,
                          mesh->o_DrT,
                          mesh->o_DsT,
                          mesh->o_q,
                          mesh->o_pmlqx,
                          mesh->o_pmlqy,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);


  if(strstr(options, "CUBATURE")){ 

    if (mesh->nonPmlNelements)
        mesh->relaxationKernel(mesh->nonPmlNelements,
                            mesh->o_nonPmlElementIds,
                            mesh->Nrhs,
                            mesh->shiftIndex,
                            mesh->o_cubInterpT,
                            mesh->o_cubProjectT,
                            mesh->o_q,
                            mesh->o_rhsq);  

    if (mesh->pmlNelements)
     mesh->pmlRelaxationKernel(mesh->pmlNelements,
                                mesh->o_pmlElementIds,
                                mesh->o_pmlIds,
                                mesh->Nrhs,
                                mesh->shiftIndex,
                                mesh->o_cubInterpT,
                                mesh->o_cubProjectT,
                                mesh->o_q,
                                mesh->o_rhsq);

  }



  if(mesh->totalHaloPairs>0){
    #if ASYNC 
      mesh->device.setStream(dataStream);
    #endif

    //make sure the async copy is finished
    mesh->device.finish();

    // start halo exchange
    meshHaloExchangeStart(mesh,
                          mesh->Nfields*mesh->Np*sizeof(dfloat),
                          sendBuffer,
                          recvBuffer);

    // wait for halo data to arrive
    meshHaloExchangeFinish(mesh);


    // copy halo data to DEVICE
    size_t offset = mesh->Np*mesh->Nfields*mesh->Nelements*sizeof(dfloat); // offset for halo data
    mesh->o_q.asyncCopyFrom(recvBuffer, haloBytes, offset);
    mesh->device.finish();        

    #if ASYNC 
      mesh->device.setStream(defaultStream);
    #endif
  }


// printf("Starting Surface step: mesh Nrhs: %d  %d \n", mesh->Nrhs, mesh->shiftIndex);

  if (mesh->nonPmlNelements)
    mesh->surfaceKernel(mesh->nonPmlNelements,
                        mesh->o_nonPmlElementIds,
                        t,
                        ramp,
                        mesh->Nrhs,
                        mesh->shiftIndex,
                        mesh->o_sgeo,
                        mesh->o_LIFTT,
                        mesh->o_vmapM,
                        mesh->o_vmapP,
                        mesh->o_EToB,
                        mesh->o_x,
                        mesh->o_y,
                        mesh->o_q,
                        mesh->o_rhsq);

  if (mesh->pmlNelements)
    mesh->pmlSurfaceKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          t,
                          ramp,
                          mesh->Nrhs,
                          mesh->shiftIndex,
                          mesh->o_sgeo,
                          mesh->o_LIFTT,
                          mesh->o_vmapM,
                          mesh->o_vmapP,
                          mesh->o_EToB,
                          mesh->o_x,
                          mesh->o_y,
                          mesh->o_q,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);


  mesh->shiftIndex = (mesh->shiftIndex+1)%3;


   // LSERK4 stages
  for(iint rk=0;rk<mesh->Nrk;++rk){

    // intermediate stage time
    dfloat t = mesh->startTime + tstep*mesh->dt + mesh->dt*mesh->rkc[rk];
    
   if(mesh->totalHaloPairs>0){
    // extract halo on DEVICE
    #if ASYNC 
      mesh->device.setStream(dataStream);
    #endif

    iint Nentries = mesh->Np*mesh->Nfields;
    mesh->haloExtractKernel(mesh->totalHaloPairs,
                            Nentries,
                            mesh->o_haloElementList,
                            mesh->o_q,
                            mesh->o_haloBuffer);

    // copy extracted halo to HOST
    mesh->o_haloBuffer.asyncCopyTo(sendBuffer);

    #if ASYNC 
      mesh->device.setStream(defaultStream);
    #endif
  }

    

    // COMPUTE RAMP FUNCTION 
    dfloat ramp, drampdt;
    boltzmannRampFunction2D(t, &ramp, &drampdt);
      

    // VOLUME KERNELS
    mesh->device.finish();
    occa::tic("volumeKernel");
    
    // compute volume contribution to DG boltzmann RHS
    if(mesh->pmlNelements){ 
      mesh->device.finish();
       occa::tic("PML_volumeKernel");

    mesh->pmlVolumeKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          ramp, 
                          drampdt,
                          one,
                          zero,
                          mesh->o_vgeo,
                          mesh->o_pmlSigmaX,
                          mesh->o_pmlSigmaY,
                          mesh->o_DrT,
                          mesh->o_DsT,
                          mesh->o_q,
                          mesh->o_pmlqx,
                          mesh->o_pmlqy,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);
      //
       mesh->device.finish();
      occa::toc("PML_volumeKernel");  

    }

    // compute volume contribution to DG boltzmann RHS added d/dt (ramp(qbar)) to RHS
    if(mesh->nonPmlNelements){

      mesh->device.finish();
      occa::tic("NONPML_volumeKernel"); 

       mesh->volumeKernel(mesh->nonPmlNelements,
                      mesh->o_nonPmlElementIds,
                      ramp, 
                      drampdt,
                      one,
                      zero,
                      mesh->o_vgeo,
                      mesh->o_DrT,
                      mesh->o_DsT,
                      mesh->o_q,
                      o_rhsqtmp);

      mesh->device.finish();
      occa::toc("NONPML_volumeKernel");

  }
    
    
    mesh->device.finish();
    occa::toc("volumeKernel");

      

      

  if(strstr(options, "CUBATURE")){ 
  // VOLUME KERNELS
    mesh->device.finish();
    occa::tic("relaxationKernel");
    // compute relaxation terms using cubature integration
    if(mesh->pmlNelements){
      mesh->device.finish();
           occa::tic("PML_relaxationKernel");

      mesh->pmlRelaxationKernel(mesh->pmlNelements,
                                mesh->o_pmlElementIds,
                                mesh->o_pmlIds,
                                one,
                                zero,
                                mesh->o_cubInterpT,
                                mesh->o_cubProjectT,
                                mesh->o_q,
                                mesh->o_rhsq);

       mesh->device.finish();
           occa::toc("PML_relaxationKernel");
    }

    // compute relaxation terms using cubature
    if(mesh->nonPmlNelements){
      mesh->device.finish();
           occa::tic("NONPML_relaxationKernel");
            
      mesh->relaxationKernel(mesh->nonPmlNelements,
                            mesh->o_nonPmlElementIds,
                            one,
                            zero,
                            mesh->o_cubInterpT,
                            mesh->o_cubProjectT,
                            mesh->o_q,
                            o_rhsqtmp);  

        mesh->device.finish();
           occa::toc("NONPML_relaxationKernel");            
    }
     // VOLUME KERNELS
    mesh->device.finish();
    occa::toc("relaxationKernel");
  }

   


  if(mesh->totalHaloPairs>0){
    #if ASYNC 
      mesh->device.setStream(dataStream);
    #endif

    //make sure the async copy is finished
    mesh->device.finish();

    // start halo exchange
    meshHaloExchangeStart(mesh,
                          mesh->Nfields*mesh->Np*sizeof(dfloat),
                          sendBuffer,
                          recvBuffer);

    // wait for halo data to arrive
    meshHaloExchangeFinish(mesh);


    // copy halo data to DEVICE
    size_t offset = mesh->Np*mesh->Nfields*mesh->Nelements*sizeof(dfloat); // offset for halo data
    mesh->o_q.asyncCopyFrom(recvBuffer, haloBytes, offset);
    mesh->device.finish();        

    #if ASYNC 
      mesh->device.setStream(defaultStream);
    #endif
  }



    // SURFACE KERNELS
    mesh->device.finish();
    occa::tic("surfaceKernel");

     if(mesh->pmlNelements){
     
      mesh->device.finish();
    occa::tic("PML_surfaceKernel"); 

     mesh->pmlSurfaceKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          t,
                          ramp,
                          one,
                          zero,
                          mesh->o_sgeo,
                          mesh->o_LIFTT,
                          mesh->o_vmapM,
                          mesh->o_vmapP,
                          mesh->o_EToB,
                          mesh->o_x,
                          mesh->o_y,
                          mesh->o_q,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);

    mesh->device.finish();
    occa::toc("PML_surfaceKernel"); 
    }
    
    if(mesh->nonPmlNelements){
        mesh->device.finish();
    occa::tic("NONPML_surfaceKernel"); 

       mesh->surfaceKernel(mesh->nonPmlNelements,
                        mesh->o_nonPmlElementIds,
                        t,
                        ramp,
                        one,
                        zero,
                        mesh->o_sgeo,
                        mesh->o_LIFTT,
                        mesh->o_vmapM,
                        mesh->o_vmapP,
                        mesh->o_EToB,
                        mesh->o_x,
                        mesh->o_y,
                        mesh->o_q,
                        o_rhsqtmp);

       mesh->device.finish();
    occa::toc("NONPML_surfaceKernel"); 
    }
    
    mesh->device.finish();
    occa::toc("surfaceKernel");
    
    // ramp function for flow at next RK stage
    dfloat tupdate = tstep*mesh->dt + mesh->dt*mesh->rkc[rk+1];
    dfloat rampUpdate, drampdtUpdate;
    boltzmannRampFunction2D(tupdate, &rampUpdate, &drampdtUpdate);
    //rampUpdate = 1.0f; drampdtUpdate = 0.0f;
   
    //printf("running with %d pml Nelements\n",mesh->pmlNelements);    
    if (mesh->pmlNelements){   
            
      mesh->RKpmlUpdateKernel(mesh->pmlNelements,
          mesh->o_pmlElementIds,
          mesh->o_pmlIds,
          mesh->dt,
          mesh->rka[rk],
          mesh->rkb[rk],
          rampUpdate,
          mesh->o_rhsq,
          mesh->o_pmlrhsqx,
          mesh->o_pmlrhsqy,
          mesh->o_resq,
          mesh->o_pmlresqx,
          mesh->o_pmlresqy,
          mesh->o_pmlqx,
          mesh->o_pmlqy,
          mesh->o_q);
   }
    
    if(mesh->nonPmlNelements){
      mesh->RKupdateKernel(mesh->nonPmlNelements,
       mesh->o_nonPmlElementIds,
       mesh->dt,
       mesh->rka[rk],
       mesh->rkb[rk],
       o_rhsqtmp,
       mesh->o_resq,
       mesh->o_q);
       mesh->device.finish();
    }

 }




}

else{
  // printf("Starting SRAB step \n");
  iint mrab_order = 2;  // Third order
  dfloat a1, a2, a3;
  if (tstep==0) {
    mrab_order = 0;  
    a1 = 1.0*mesh->dt;
    a2 = 0.0;
    a3 = 0.0;
  } else if (tstep ==1) {
    mrab_order = 1;  
    a1 =  3.0/2.0*mesh->dt;
    a2 = -1.0/2.0*mesh->dt;
    a3 =  0.0;
  } else {
  a1 =  23./12.0*mesh->dt;
  a2 = -16./12.0*mesh->dt;
  a3 =   5./12.0*mesh->dt;
  }

  // intermediate stage time
  dfloat t = mesh->startTime + (tstep+1)*mesh->dt;

  // COMPUTE RAMP FUNCTION 
  dfloat ramp, drampdt;
  boltzmannRampFunction2D(t, &ramp, &drampdt);


  if(mesh->totalHaloPairs>0){
    // extract halo on DEVICE
    #if ASYNC 
      mesh->device.setStream(dataStream);
    #endif

    iint Nentries = mesh->Np*mesh->Nfields;
    mesh->haloExtractKernel(mesh->totalHaloPairs,
                            Nentries,
                            mesh->o_haloElementList,
                            mesh->o_q,
                            mesh->o_haloBuffer);

    // copy extracted halo to HOST
    mesh->o_haloBuffer.asyncCopyTo(sendBuffer);

    #if ASYNC 
      mesh->device.setStream(defaultStream);
    #endif
  }


  if (mesh->nonPmlNelements)  
      mesh->volumeKernel(mesh->nonPmlNelements,
                      mesh->o_nonPmlElementIds,
                      ramp, 
                      drampdt,
                      mesh->Nrhs,
                      mesh->shiftIndex,
                      mesh->o_vgeo,
                      mesh->o_DrT,
                      mesh->o_DsT,
                      mesh->o_q,
                      mesh->o_rhsq);

  if (mesh->pmlNelements)
     mesh->pmlVolumeKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          ramp, 
                          drampdt,
                          mesh->Nrhs,
                          mesh->shiftIndex,
                          mesh->o_vgeo,
                          mesh->o_pmlSigmaX,
                          mesh->o_pmlSigmaY,
                          mesh->o_DrT,
                          mesh->o_DsT,
                          mesh->o_q,
                          mesh->o_pmlqx,
                          mesh->o_pmlqy,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);


  if(strstr(options, "CUBATURE")){ 

    if (mesh->nonPmlNelements)
        mesh->relaxationKernel(mesh->nonPmlNelements,
                            mesh->o_nonPmlElementIds,
                            mesh->Nrhs,
                            mesh->shiftIndex,
                            mesh->o_cubInterpT,
                            mesh->o_cubProjectT,
                            mesh->o_q,
                            mesh->o_rhsq);  

    if (mesh->pmlNelements)
     mesh->pmlRelaxationKernel(mesh->pmlNelements,
                                mesh->o_pmlElementIds,
                                mesh->o_pmlIds,
                                mesh->Nrhs,
                                mesh->shiftIndex,
                                mesh->o_cubInterpT,
                                mesh->o_cubProjectT,
                                mesh->o_q,
                                mesh->o_rhsq);

  }



  if(mesh->totalHaloPairs>0){
    #if ASYNC 
      mesh->device.setStream(dataStream);
    #endif

    //make sure the async copy is finished
    mesh->device.finish();

    // start halo exchange
    meshHaloExchangeStart(mesh,
                          mesh->Nfields*mesh->Np*sizeof(dfloat),
                          sendBuffer,
                          recvBuffer);

    // wait for halo data to arrive
    meshHaloExchangeFinish(mesh);


    // copy halo data to DEVICE
    size_t offset = mesh->Np*mesh->Nfields*mesh->Nelements*sizeof(dfloat); // offset for halo data
    mesh->o_q.asyncCopyFrom(recvBuffer, haloBytes, offset);
    mesh->device.finish();        

    #if ASYNC 
      mesh->device.setStream(defaultStream);
    #endif
  }


// printf("Starting Surface step: mesh Nrhs: %d  %d \n", mesh->Nrhs, mesh->shiftIndex);

  if (mesh->nonPmlNelements)
    mesh->surfaceKernel(mesh->nonPmlNelements,
                        mesh->o_nonPmlElementIds,
                        t,
                        ramp,
                        mesh->Nrhs,
                        mesh->shiftIndex,
                        mesh->o_sgeo,
                        mesh->o_LIFTT,
                        mesh->o_vmapM,
                        mesh->o_vmapP,
                        mesh->o_EToB,
                        mesh->o_x,
                        mesh->o_y,
                        mesh->o_q,
                        mesh->o_rhsq);

  if (mesh->pmlNelements)
    mesh->pmlSurfaceKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          t,
                          ramp,
                          mesh->Nrhs,
                          mesh->shiftIndex,
                          mesh->o_sgeo,
                          mesh->o_LIFTT,
                          mesh->o_vmapM,
                          mesh->o_vmapP,
                          mesh->o_EToB,
                          mesh->o_x,
                          mesh->o_y,
                          mesh->o_q,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy);



// printf("Starting update step \n");


  const iint id = mrab_order*3;
  // mesh->MRAB_C[0],
                      // mesh->MRAB_A[id+0], //
                      // mesh->MRAB_A[id+1],
                      // mesh->MRAB_A[id+2], //

  if (mesh->nonPmlNelements)
    mesh->updateKernel(mesh->nonPmlNelements,
                      mesh->o_nonPmlElementIds,
                      mesh->dt,
                      a1,a2,a3,
                      mesh->shiftIndex,
                      mesh->o_rhsq,
                      mesh->o_q);

  if (mesh->pmlNelements)
    mesh->pmlUpdateKernel(mesh->pmlNelements,
                          mesh->o_pmlElementIds,
                          mesh->o_pmlIds,
                          mesh->dt,
                          a1,a2,a3,
                          mesh->shiftIndex,
                          mesh->o_rhsq,
                          mesh->o_pmlrhsqx,
                          mesh->o_pmlrhsqy,
                          mesh->o_q,
                          mesh->o_pmlqx,
                          mesh->o_pmlqy);

  //rotate index
  mesh->shiftIndex = (mesh->shiftIndex+1)%3;


}


}
