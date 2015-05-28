// Filename: loadDataMain.cu
//
// Copyright (c) 2010-2012, Florencio Balboa Usabiaga
//
// This file is part of Fluam
//
// Fluam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Fluam is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Fluam. If not, see <http://www.gnu.org/licenses/>.


#define GLOBALS_CELLS 1
#define GLOBALS_FLUID 1        
#define GLOBALS_PARTICLES 1
#define GLOBALS_BOUNDARY 1

#include "header.h"
#include "cells.h"
#include "fluid.h"
#include "particles.h"
#include "boundary.h"

//Other functions
#define GLOBALS_OTHER_FLUID_V 1
#include "headerOtherFluidVariables.h"

//Parameters
//NEW_PARAMETER
const string widentity_prefactor="identity_prefactor";
const string wcontinuousGradient="continuousGradient";
const string wgradTemperature="gradTemperature";
const string wsoretCoefficient="soretCoefficient";
const string wbackgroundvelocity="backgroundvelocity";
const string wsetDevice="setDevice";
const string wsetparticles="particles";
const string wparticlesdata="particlesdata";
string particlesdata;
const string wsetboundary="boundary";
const string wfileboundary="fileboundary";
const string wvelocityboundary="velocityboundary";
int createboundary;
const string wvolumeboundaryconst="volumeboundaryconst";
const string wdensfluid="densfluid";
const string wshearviscosity="shearviscosity";
const string wbulkviscosity="bulkviscosity";
const string wtemperature="temperature";
const string wthermostat="thermostat";
const string wnumsteps="numsteps";
const string wnumstepsRelaxation="numstepsRelaxation";
const string wnothing="#";
const string wdt="dt";
const string wsamplefreq="samplefreq";
const string wsavefreq="savefreq";
const string wpressureparameters="pressureparameters";
const string woutputname="outputname";
const string wseed="seed";
const string wcells="cells";
const string wcelldimension="celldimension";
const string winitfluid="initfluid";
const string wsavedensity="savedensity";
const string wloadparticles="loadparticles";
const string wparticlescoor="coordinates";
const string wparticlesvel="velocities";
const string wloadFluid="fluid";
const string wnumberparticles="numberparticles";
const string wmass="mass";
const string wvolumeParticle="volumeParticle";
const string wcutoff="cutoff";
const string wmaxNumberPartInCell="maxNumberPartInCell";
const string wmaxNumberPartInCellNonBonded="maxNumberPartInCellNonBonded";
const string wdDensity="dDensity";
const string womega="omega";
const string wwaveSourceIndex="waveSourceIndex";
const string wCheckVelocity="checkVelocity";
const string wsaveFluid="saveFluid";
const string wsaveVTK="saveVTK";
const string wbondedForces="bondedForces";
const string wcomputeNonBondedForces="computeNonBondedForces";

const string wGhost="ghost";
//Other Fluid Variables
//Binary Mixture Begins
const string wbinaryMixture="binaryMixture";
const string wdiffusion="diffusion";
const string wmassSpecies0="massSpecies0";
const string wmassSpecies1="massSpecies1";
const string wfileConcentration="fileConcentration";
string fileConcentration;
const string wconcentration="concentration";
//Binary Mixture Ends
//Binary Mixture Wall Begins
const string wbinaryMixtureWall="binaryMixtureWall";
const string wconcentrationWall="concentrationWall";
const string wvxWall="vxWall";
const string wvyWall="vyWall";
const string wvzWall="vzWall";
const string wdensityWall="densityWall";
//Binary Mixture Wall Ends  
//Giant Fluctuation Begins
const string wgiantFluctuations="giantFluctuations";
//Giant Fluctuation Ends
//Incompressible Begins
const string wincompressible="incompressible";
//Incompressible ends
//Incompressible Buondary RK2 Begins
const string wincompressibleBoundaryRK2="incompressibleBoundaryRK2";
//Incompressible Buondary RK2 Ends
//quasiNeutrallyBuoyant Begins
const string wquasiNeutrallyBuoyant="quasiNeutrallyBuoyant";
//quasiNeutrallyBuoyant Ends
//quasiNeutrallyBuoyant2D Begins
const string wquasiNeutrallyBuoyant2D="quasiNeutrallyBuoyant2D";
//quasiNeutrallyBuoyant2D Ends
const string wquasiNeutrallyBuoyant4pt2D="quasiNeutrallyBuoyant4pt2D";
//quasiNeutrallyBuoyant4pt2D Ends
//quasiNeutrallyBuoyant4pt2D Begins
//IMEX-RK Begins
const string wIMEXRK="IMEXRK";
//IMEX-RK Ends
//IncompressibleBinaryMixture Begins
const string wincompressibleBinaryMixture="incompressibleBinaryMixture";
//IncompressibleBinaryMixture Ends
//IncompressibleBinaryMixtureMidPoint Begins 
const string wincompressibleBinaryMixtureMidPoint="incompressibleBinaryMixtureMidPoint";
//IncompressibleBinaryMixtureMidPoint Ends
//particlesWall Begins
const string wparticlesWall="particlesWall";
//particlesWall Ends
//freeEnergyCompressibleParticles Begins
const string wfreeEnergyCompressibleParticles="freeEnergyCompressibleParticles";
const string womega0="omega0";
//freeEnergyCompressibleParticles Ends
//semiImplicitCompressibleParticles Begins
const string wsemiImplicitCompressibleParticles="semiImplicitCompressibleParticles";
//semiImplicitCompressibleParticles Ends
//momentumCoupling Begins
const string wmomentumCoupling="momentumCoupling";
//momentumCoupling Ends
//stokesLimit Begins
const string wstokesLimit="stokesLimit";
const string wextraMobility="extraMobility";
//stokesLimit Ends


bool loadDataMain(int argc, char* argv[]){

  //DEFAULT PARAMETERS
  //NEW_PARAMETER
  identity_prefactor=1.;
  step = 0;
  savefreq=0;
  setDevice=-1;
  bool_seed=0;
  setparticles = 0;
  setboundary=0;
  velocityboundary=0;
  thermostat=0;
  samplefreq = 0;
  savedensity = 0;
  volumeboundaryconst = 8.;
  loadparticles=0;
  maxNumberPartInCell = 6;
  maxNumberPartInCellNonBonded = 6;
  cutoff = 0.;
  waveSourceIndex = 0;
  loadFluid=0;
  setCheckVelocity = 0;
  setSaveFluid = 0;
  setSaveVTK=0;
  initfluid = 0;
  numstepsRelaxation = 0;
  setGhost=0;
  vx0=0; vy0=0; vz0=0;
  np = 0;
  mass = 0;
  nboundary = 0;
  bondedForces=0;
  computeNonBondedForces=1;
  //DEFAULT PARAMETERS 

  //OTHER FLUID VARIABLES
  //Binary Mixture Begins
   setBinaryMixture = 0;
  diffusion = 0; //Mass diffusion for a binary mixture
  massSpecies0 = 0;
  massSpecies1 = 0;
  concentration=0;
  //Binary Mixture Ends

  //Binary Mixture Wall Begins
  setBinaryMixtureWall = 0;
  vxWall0 = 0;
  vxWall1 = 0;
  vzWall0 = 0;
  vzWall1 = 0;
  //Binary Mixture Wall Ends

  //Giant Fluctiations Begins
  setGiantFluctuations = 0;
  //Giant Fluctuations Ends

  //Continuous Gradient Begins
  setContinuousGradient = 0;
  //Continuous Gradient Ends

  //Incompressible Begins
  incompressible = 0;
  //Incompressible Ends

  //Incompressible Boundary RK2 Begins
  incompressibleBoundaryRK2 = 0;
  //Incompressible Boundary RK2 Ends

  //quasiNeutrallyBuoyant Begins
  quasiNeutrallyBuoyant = 0;
  //quasiNeutrallyBuoyant Ends

  //quasiNeutrallyBuoyant2D Begins
  quasiNeutrallyBuoyant2D = 0;
  //quasiNeutrallyBuoyant2D Ends

  //quasiNeutrallyBuoyant4pt2D Begins
  quasiNeutrallyBuoyant4pt2D = 0;
  //quasiNeutrallyBuoyant4pt2D Ends

  //particlesWall Begins
  particlesWall = 0;
  //particlesWall Ends

  //freeEnergyCompressibleParticles Begins
  freeEnergyCompressibleParticles = 0;
  omega0 = 0;
  //freeEnergyCompressibleParticles Ends

  //semiImplicitCompressibleParticles Begins
  semiImplicitCompressibleParticles = 0;
  //semiImplicitCompressibleParticles Ends

  //momentumCoupling Begins
  momentumCoupling = 0;
  //momentumCoupling Ends

  //stokesLimit Begins
  stokesLimit = 0;
  setExtraMobility = 0;
  extraMobility = 0;
  //stokesLimit Ends


  ifstream fileinput, fileoutput;
  string fileinputname;
  if(argc!=1){
    fileinputname=argv[1];
  }
  else if(argc==1){
    fileinputname="data.main";
  }
  fileinput.open(fileinputname.c_str());


  string word, oldword, wordfile;
  //Copy data.main to wordfile
  while(!fileinput.eof()){
    getline(fileinput,word);
    wordfile += word + "\n";
  }
  fileinput.close();
  //Open again data.main and read parameters for the simulation
  fileinput.open(fileinputname.c_str());

  fileinput >> word;
  while(!fileinput.eof()){
    
    if(word==wsetparticles){
      fileinput >> setparticles;
    }
    //NEW_PARAMETER
    else if(word==widentity_prefactor){
      fileinput >> identity_prefactor;
    }
    else if(word==wgradTemperature){
      fileinput >> gradTemperature;
    }
    else if(word==wsoretCoefficient){
      fileinput >> soretCoefficient;
    }
    else if(word==wbackgroundvelocity){
      fileinput >> vx0 >> vy0 >> vz0;
    }
    else if(word==wsetDevice){
      fileinput >> setDevice;
    }
    else if(word==wparticlesdata){
      fileinput >> particlesdata;
    }
    else if(word==wsetboundary){
      setboundary = 1;
      fileinput >> fileboundary;
    }
    else if(word==wvelocityboundary){
      fileinput >> velocityboundary;
    }
    else if(word==wvolumeboundaryconst){
      fileinput >> volumeboundaryconst;
    }
    else if(word==wdensfluid){
      fileinput >> densfluid;
      densityConst = densfluid;
    }
    else if(word==wshearviscosity){
      fileinput >> shearviscosity;
    }
    else if(word==wbulkviscosity){
      fileinput >> bulkviscosity;
    }
    else if(word==wtemperature){
      fileinput >> temperature;
    }
    else if(word==wthermostat){
      thermostat = 1;
    }
    else if(word==wnumsteps){
      fileinput >> numsteps;
    }
    else if(word==wnumstepsRelaxation){
      fileinput >> numstepsRelaxation;
    }
    else if(word==wdt){
      fileinput >> dt;
    }
    else if(word==wsamplefreq){
      fileinput >> samplefreq;
    }
    else if(word==wsavefreq){
      fileinput >> savefreq;
    }
    else if(word==wpressureparameters){
      fileinput >> pressurea0 >> pressurea1 >> pressurea2;
    }
    else if(word==woutputname){
      fileinput >> outputname;
    }
    else if(word==wseed){
      fileinput >> seed;
      bool_seed=1;
    }
    else if(word==wcells){
      fileinput >> mx >> my >> mz;
    }
    else if(word==wcelldimension){
      fileinput >> lx >> ly >> lz;
    }
    else if(word==winitfluid){
      fileinput >> initfluid;
    }
    else if(word==wsavedensity){
      fileinput >> savedensity;
      if(savedensity==1){
	cout << "READING SAVEDENSITY" << endl;
	fileinput >> nsavedensity;
	cellsavedensity = new int [nsavedensity];
	for(int i=0;i<nsavedensity;i++)
	  fileinput >> cellsavedensity[i];
      }
      else getline(fileinput,word);
    }
    else if(word==wCheckVelocity){
      setCheckVelocity = 1;
      fileinput >> fileCheckVelocity;
    }
    else if(word==wloadparticles){
      fileinput >> loadparticles;
    }
    else if(word==wparticlescoor){
      fileinput >> particlescoor;
    }
    else if(word==wparticlesvel){
      fileinput >> particlesvel;
    }
    else if(word==wloadFluid){
      fileinput >> loadFluidFile;
      loadFluid=1;
    }
    else if(word==wnumberparticles){
      fileinput >> np;
    }
    else if(word==wmass){
      fileinput >> mass;
    }
    else if(word==wvolumeParticle){
      fileinput >> volumeParticle;
      //volumeParticle = volumeParticle/c[0].volume;
    }
    else if(word==wcutoff){
      fileinput >> cutoff;
    }
    else if(word==wmaxNumberPartInCell){
      fileinput >> maxNumberPartInCell;
    }
    else if(word==wmaxNumberPartInCellNonBonded){
      fileinput >> maxNumberPartInCellNonBonded;
    }
    else if(word==wdDensity){
      fileinput >> dDensity;
    }
    else if(word==womega){
      fileinput >> omega;
    }
    else if(word==wwaveSourceIndex){
      fileinput >> waveSourceIndex;
    }
    else if(word==wsaveFluid){
      fileinput >> setSaveFluid;
    }
    else if(word==wsaveVTK){
      fileinput >> setSaveVTK;
    }
    else if(word==wGhost){
      setGhost = 1;
    }
    else if(word==wbondedForces){
      bondedForces=1;
      fileinput >> bondedForcesFile;
    }
    else if(word==wcomputeNonBondedForces){
      fileinput >> computeNonBondedForces;
    }
    //Other FLuid Variables
    //Binary Mixture Begins
    else if(word==wbinaryMixture){
      setBinaryMixture = 1;
    }
    else if(word==wdiffusion){
      fileinput >> diffusion;
    }
    else if(word==wmassSpecies0){
      fileinput >> massSpecies0;
    }
    else if(word==wmassSpecies1){
      fileinput >> massSpecies1;
    }
    else if(word==wconcentration){
      fileinput >> concentration;
    }
    //Binary Mixture Ends
    //Binary Mixture Wall Begins
    else if(word==wbinaryMixtureWall){
      setBinaryMixtureWall = 1;
    }
    else if(word==wconcentrationWall){
      fileinput >> cWall0 >> cWall1;
    }
    else if(word==wvxWall){
      fileinput >> vxWall0 >> vxWall1;
    }
    else if(word==wvyWall){
      fileinput >> vyWall0 >> vyWall1;
    }
    else if(word==wvzWall){
      fileinput >> vzWall0 >> vzWall1;
    }
    else if(word==wdensityWall){
      fileinput >> densityWall0 >> densityWall1;
    }
    //Binary Mixture Wall Ends
    //Giant Fluctuations Bigins
    else if(word==wgiantFluctuations){
      setGiantFluctuations = 1;
    }
    //Giant Fluctuations Ends
    //Continupus Gradient Begins
    else if(word==wcontinuousGradient){
      setContinuousGradient=1;
    }
    //Continupus Gradient Ends
    //Incompressible Begins
    else if(word==wincompressible){
      incompressible=1;
    }
    //Incompressible Ends
    //Incompressible Boundary RK2 Begins
    else if(word==wincompressibleBoundaryRK2){
      incompressibleBoundaryRK2=1;
    }
    //Incompressible Boundary RK2 Ends
    //quasiNeutrallyBuoyant Begins
    else if(word==wquasiNeutrallyBuoyant){
      quasiNeutrallyBuoyant=1;
    }
    //quasiNeutrallyBuoyant Ends
    //quasiNeutrallyBuoyant2D Begins
    else if(word==wquasiNeutrallyBuoyant2D){
      quasiNeutrallyBuoyant2D=1;
    }
    //quasiNeutrallyBuoyant2D Ends
    //quasiNeutrallyBuoyant4pt2D Begins
    else if(word==wquasiNeutrallyBuoyant4pt2D){
      quasiNeutrallyBuoyant4pt2D=1;
    }
    //quasiNeutrallyBuoyant4pt2D Ends
    //IMEXRK Begins
    else if(word==wIMEXRK){
      IMEXRK=1;
      setparticles=1;
    }
    //IMEXRK Ends
    //IncompressibleBinaryMixture Begins
    else if(word==wincompressibleBinaryMixture){
      incompressibleBinaryMixture=1;
    }
    //IncompressibleBinaryMixture Ends
    //IncompressibleBinaryMixtureMidPoint Begins
    else if(word==wincompressibleBinaryMixtureMidPoint){
      incompressibleBinaryMixtureMidPoint=1;
    }
    //IncompressibleBinaryMixtureMidPoint Ends
    //particlesWall Begins
    else if(word==wparticlesWall){
      particlesWall=1;
      setparticles=1;
    }
    //particlesWall Ends
    //freeEnergyCompressibleParticles Begins
    else if(word==wfreeEnergyCompressibleParticles){
      freeEnergyCompressibleParticles=1;
      setparticles=1;
    }
    else if(word==womega0){
      fileinput >> omega0;
    }
    //freeEnergyCompressibleParticles Ends
    //semiImplicitCompressibleParticles Begins
    else if(word==wsemiImplicitCompressibleParticles){
      semiImplicitCompressibleParticles=1;
      setparticles=1;
    }
    //semiImplicitCompressibleParticles Ends
    //momentumCoupling Begins
    else if(word==wmomentumCoupling){
      momentumCoupling=1;
      setparticles=1;
    }
    //momentumCouplings Ends
    //stokesLimit Begins
    else if(word==wstokesLimit){
      stokesLimit=1;
      setparticles=1;
    }
    else if(word==wextraMobility){
      fileinput >> extraMobility;
      if(extraMobility != 0){
	setExtraMobility = 1;
      }
    }
    //stokesLimit Ends
    else if(word.substr(0,1)==wnothing){
      getline(fileinput,word);
    }
    else{
      cout << "ERROR INPUT FILE" << endl;
      cout << "LAST WORD READED: " << oldword << endl;
      return 0;
    }
    
    oldword = word;
    fileinput >> word;
  }
  fileinput.close();

  //Save wordfile
  string fileOutName;
  fileOutName = outputname + ".data.main";
  ofstream fileout(fileOutName.c_str());
  fileout << wordfile << endl;
  fileout.close();
  //cout << 




  cout <<  "READ " << fileinputname.c_str() << " :                DONE" << endl;

  return 1;
}
