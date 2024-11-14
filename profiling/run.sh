#######################################################################################
## Objectif : 		Lance plusieurs profilers sur un binaire
## Fonctionnement : 	. run.sh <binary>
## Exemple : 		. profilers/run.sh  ./build/demos/tutorial/reconstruction
#######################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



## recupérer le nom du binaire en argument CLI 
if [ -z "$1" ]; then 
	echo "ERROR : Usage : $0 <kernel_binary>"
#	exit 1
fi
KERNEL="$1"


function run_perf {
	echo "Launch PERF : ${KERNEL}."
	. ${SCRIPT_DIR}/perf.sh all ${KERNEL}
}

function run_maqao {
        echo "Launch MAQAO : ${KERNEL}."
        . ${SCRIPT_DIR}/maqao.sh oneview ${KERNEL}
}

function run_vtune {
        echo "Launch VTUNE: ${KERNEL}."
        . ${SCRIPT_DIR}/vtune.sh hotspots ${KERNEL}
}



# create directoty
# name of binary
PROFILING_DIR="profiling_results"
BINARY_NAME=$(basename ${KERNEL})
mkdir -p "${PROFILING_DIR}"
CURRENT_PATH=$(pwd)
cd ${PROFILING_DIR}
	mkdir -p ${BINARY_NAME}
	cd ${BINARY_NAME}
		mkdir -p perf
		mkdir -p maqao
		mkdir -p vtune
		cd perf
			run_perf 2>perf.txt
		cd ..
		cd maqao
			run_maqao
		cd ..
		cd vtune
			run_vtune
		cd ..
	cd ..
cd ${CURRENT_PATH}

