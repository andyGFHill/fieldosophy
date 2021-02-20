
# Variables to set for your setup
EIGENPATH = ./externals



CPP = g++	# Chosen compiler
INCLUDES = -I ./fieldosophy/Includes -I ${EIGENPATH}	# Path to local includes in c/c++ code
SRCPATH = ./fieldosophy/src
MACROSDEFINED = -DEIGEN_MPL2_ONLY  # Macros to be defined
LINKERFLAGS = -lgsl -lgslcblas -lm -lgomp -O3 	# Flags for the linker
COMPILERFLAGS = -shared -fPIC -fopenmp  	# Flags for the compiler
LIBRARYOUTPUTPATH = ./fieldosophy/libraries


# Define sources for all components
MESHSRC = ${SRCPATH}/mesh/mesh.cxx ${SRCPATH}/mesh/meshGraph.cxx ${SRCPATH}/mesh/implicitMesh.cxx ${SRCPATH}/mesh/HyperRectExtension.cxx ${SRCPATH}/mesh/MeshAndMetric.cxx
MARGINALSRC = ${SRCPATH}/marginal/NIG.c
FEMSRC = ${SRCPATH}/FEM.c

ALLSRC = ${MESHSRC} ${MARGINALSRC} ${FEMSRC} ${SRCPATH}/misc.cxx




# Different build options


all: clean Fieldosophy


clean:
	@echo ""
	@echo "Cleaning up"
	rm -f ${LIBRARYOUTPUTPATH}/*.so
	@echo ""

Fieldosophy:
	@echo "Compiling Fieldosophy library"
	${CPP} ${MACROSDEFINED} ${COMPILERFLAGS} -o ${LIBRARYOUTPUTPATH}/libSPDEC.so  ${ALLSRC}  ${INCLUDES} ${LINKERFLAGS} 
	@echo ""


Cholesky: 
	@echo "Compiling Cholesky library"
	${CPP} ${MACROSDEFINED} ${COMPILERFLAGS} -o ${LIBRARYOUTPUTPATH}/libCholesky.so ${SRCPATH}/Cholesky.c -I ${INCLUDES} ${LINKERFLAGS}
	@echo ""







