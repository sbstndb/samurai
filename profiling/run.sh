#######################################################################################
## Objectif : 		Lance plusieurs profilers sur un binaire
## Fonctionnement : 	. run.sh <binary>
## Exemple : 		. profilers/run.sh  ./build/demos/tutorial/reconstruction
#######################################################################################

## recupérer le nom du binaire en argument CLI 
if [ -z "$1" ]; then 
	echo "ERROR : Usage : $0 <kernel_binary>"
#	exit 1
fi
KERNEL="$1"


function run_perf {
	echo "Launch PERF : ${KERNEL}."
	. perf.sh all ${KERNEL}
}

function run_maqao {
        echo "Launch MAQAO : ${KERNEL}."
        . maqao.sh oneview ${KERNEL}
}

function run_vtune {
        echo "Launch VTUNE: ${KERNEL}."
        . vtune.sh ${KERNEL}
}



run_perf
run_maqao
run_vtune

