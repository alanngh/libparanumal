/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

/* wall 1, inflow 2, outflow 3 */

// Weakly Impose Nonlinear term BCs
#define insAdvectionBoundaryConditions3D(bc, t, x, y, z, nx, ny, nz, uM, vM, wM, uB, vB, wB) \
  {	\
    dfloat a = OCCA_PI/4.f; \
    dfloat d = OCCA_PI/2.f; \
    if(bc==1){								\
      *(uB) = 0.f;							\
      *(vB) = 0.f;							\
      *(wB) = 0.f;              \
    } else if(bc==2){							\
      *(uB) = -a*(occaExp(a*x)*occaSin(a*y+d*z)+occaExp(a*z)*occaCos(a*x+d*y))*occaExp(-d*d*t);\
      *(vB) = -a*(occaExp(a*y)*occaSin(a*z+d*x)+occaExp(a*x)*occaCos(a*y+d*z))*occaExp(-d*d*t);\
      *(wB) = -a*(occaExp(a*z)*occaSin(a*x+d*y)+occaExp(a*y)*occaCos(a*z+d*x))*occaExp(-d*d*t);\
    } else if(bc==3){							\
      *(uB) = uM;							\
      *(vB) = vM;							\
      *(wB) = wM;             \
    }									\
  }

#define insDivergenceBoundaryConditions3D(bc, t, x, y, z, nx, ny, nz, uM, vM, wM, uB, vB, wB) \
  {	\
    dfloat a = OCCA_PI/4.f; \
    dfloat d = OCCA_PI/2.f; \
       if(bc==1){               \
      *(uB) = 0.f;              \
      *(vB) = 0.f;              \
      *(wB) = 0.f;              \
    } else if(bc==2){             \
      *(uB) = -a*(occaExp(a*x)*occaSin(a*y+d*z)+occaExp(a*z)*occaCos(a*x+d*y))*occaExp(-d*d*t);\
      *(vB) = -a*(occaExp(a*y)*occaSin(a*z+d*x)+occaExp(a*x)*occaCos(a*y+d*z))*occaExp(-d*d*t);\
      *(wB) = -a*(occaExp(a*z)*occaSin(a*x+d*y)+occaExp(a*y)*occaCos(a*z+d*x))*occaExp(-d*d*t);\
    } else if(bc==3){             \
      *(uB) = uM;             \
      *(vB) = vM;             \
      *(wB) = wM;             \
    }                 \
  }

// Gradient only applies to Pressure and Pressure Incremament
// Boundary Conditions are implemented in strong form
#define insGradientBoundaryConditions3D(bc,t,x,y,z,nx,ny,nz,pM,pB)	\
  {	\
    dfloat a = OCCA_PI/4.f; \
    dfloat d = OCCA_PI/2.f; \
    if(bc==1){							\
      *(pB) = pM;						\
    } else if(bc==2){						\
      *(pB) = pM;						\
    } else if(bc==3){						\
      *(pB) = -a*a*occaExp(-2.f*d*d*t)*(occaExp(2.f*a*x)+occaExp(2.f*a*y)+occaExp(2.f*a*z))*(occaSin(a*x+d*y)*occaCos(a*z+d*x)*occaExp(a*(y+z))+occaSin(a*y+d*z)*occaCos(a*x+d*y)*occaExp(a*(x+z))+occaSin(a*z+d*x)*occaCos(a*y+d*z)*occaExp(a*(x+y))); \
    }								\
  }

#define insHelmholtzBoundaryConditionsIpdg3D(bc,t,x,y,z, nx,ny,nz, uB,uxB,uyB,uzB, vB,vxB,vyB,vzB, wB,wxB,wyB,wzB) \
  {	dfloat a = OCCA_PI/4.f; \
    dfloat d = OCCA_PI/2.f; \
    if((bc==1)||(bc==4)){						\
      *(uB) = 0.f;							\
      *(vB) = 0.f;              \
      *(wB) = 0.f;              \
									              \
      *(uxB) = 0.f;							\
      *(uyB) = 0.f;             \
      *(uzB) = 0.f;             \
                                \
      *(vxB) = 0.f;							\
      *(vyB) = 0.f;             \
      *(vzB) = 0.f;             \
                                \
      *(wxB) = 0.f;             \
      *(wyB) = 0.f;             \
      *(wzB) = 0.f;             \
    } else if(bc==2){							\
      \
      *(uB) = -a*(occaExp(a*x)*occaSin(a*y+d*z)+occaExp(a*z)*occaCos(a*x+d*y))*occaExp(-d*d*t); \
      *(vB) = -a*(occaExp(a*y)*occaSin(a*z+d*x)+occaExp(a*x)*occaCos(a*y+d*z))*occaExp(-d*d*t); \
      *(wB) = -a*(occaExp(a*z)*occaSin(a*x+d*y)+occaExp(a*y)*occaCos(a*z+d*x))*occaExp(-d*d*t); \
									\
      *(uxB) = 0.f;             \
      *(uyB) = 0.f;             \
      *(uzB) = 0.f;             \
                                \
      *(vxB) = 0.f;             \
      *(vyB) = 0.f;             \
      *(vzB) = 0.f;             \
                                \
      *(wxB) = 0.f;             \
      *(wyB) = 0.f;             \
      *(wzB) = 0.f;             \
    } else if(bc==3){							\
      *(uB) = 0.f;   \
      *(vB) = 0.f;   \
      *(wB) = 0.f;   \
                  \
      *(uxB) = -a*(a*occaExp(a*x)*occaSin(a*y+d*z)-a*occaExp(a*z)*occaSin(a*x+d*y))*occaExp(-d*d*t); \
      *(uyB) = -a*(a*occaExp(a*x)*occaCos(a*y+d*z)-d*occaExp(a*z)*occaSin(a*x+d*y))*occaExp(-d*d*t); \
      *(uzB) = -a*(d*occaExp(a*x)*occaCos(a*y+d*z)+a*occaExp(a*z)*occaCos(a*x+d*y))*occaExp(-d*d*t); \
                                \
      *(vxB) = -a*(d*occaExp(a*y)*occaCos(a*z+d*x)+a*occaExp(a*x)*occaCos(a*y+d*z))*occaExp(-d*d*t); \
      *(vyB) = -a*(a*occaExp(a*y)*occaSin(a*z+d*x)-a*occaExp(a*x)*occaSin(a*y+d*z))*occaExp(-d*d*t); \
      *(vzB) = -a*(a*occaExp(a*y)*occaCos(a*z+d*x)-d*occaExp(a*x)*occaSin(a*y+d*z))*occaExp(-d*d*t); \
                                \
      *(wxB) = a*(a*occaExp(a*z)*occaCos(a*x+d*y)-d*occaExp(a*y)*occaSin(a*z+d*x))*occaExp(-d*d*t); \
      *(wyB) = a*(d*occaExp(a*z)*occaCos(a*x+d*y)+a*occaExp(a*y)*occaCos(a*z+d*x))*occaExp(-d*d*t); \
      *(wzB) = a*(a*occaExp(a*z)*occaSin(a*x+d*y)-a*occaExp(a*y)*occaSin(a*z+d*x))*occaExp(-d*d*t); \
    }									\
  }
// Give dudx on bc==3

// Compute bcs for P increment
#define insPoissonBoundaryConditions3D(bc,t,dt,x,y,z,nx,ny,nz,pB,pxB,pyB,pzB)	\
  {	\
    dfloat a = OCCA_PI/4.f; \
    dfloat d = OCCA_PI/2.f; \
    if((bc==1)||(bc==4)){						\
      *(pB) = 0.f;							\
									\
      *(pxB) = 0.f;							\
      *(pyB) = 0.f;							\
      *(pzB) = 0.f;              \
    }									\
    if(bc==2){								\
      *(pB)  = 0.f;							\
									\
      *(pxB) = 0.f; \
      *(pyB) = 0.f; \
      *(pzB) = 0.f; \
    }									\
    if(bc==3){								\
      *(pB) = -a*a*occaExp(-2.f*d*d*t)*( occaExp(2.f*a*x)+occaExp(2.f*a*y)+occaExp(2.f*a*z))*(occaSin(a*x+d*y)*occaCos(a*z+d*x)*occaExp(a*(y+z))+occaSin(a*y+d*z)*occaCos(a*x+d*y)*occaExp(a*(x+z))+occaSin(a*z+d*x)*occaCos(a*y+d*z)*occaExp(a*(x+y))); \
									\
      *(pxB) = 0.f;							\
      *(pyB) = 0.f;							\
      *(pzB) = 0.f;             \
    }									\
  }

  // Compute bcs for P increment
#define insPoissonNeumannTimeDerivative3D(bc,t,x,y,z,dpdt)  \
  { \
    if((bc==1)||(bc==4)||(bc==2) ){           \
      *(dpdt) = 2.f*d*d*a*a*occaExp(-2.f*d*d*t)*( occaExp(2.f*a*x)+occaExp(2.f*a*y)+occaExp(2.f*a*z))*(occaSin(a*x+d*y)*occaCos(a*z+d*x)*occaExp(a*(y+z))+occaSin(a*y+d*z)*occaCos(a*x+d*y)*occaExp(a*(x+z))+occaSin(a*z+d*x)*occaCos(a*y+d*z)*occaExp(a*(x+y))); \
    }                 \
    if(bc==3){                \
      *(dpdt) = 0.f; \
    }                 \
  }