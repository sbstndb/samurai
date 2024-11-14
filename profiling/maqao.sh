#######################################################################################
## Objectif : 		pouvoir facilement lancer les commandes MAQAO classiques
## Outil MAQAO : 	permet d'obtenir des indicateurs de performance 
##				(vectorisation, branchement, ...)
## Fonctionnement : 	. maqao.sh <mode> <binary>
## Exemple : 		. profilers/maqao.sh all ./build/demos/tutorial/reconstruction
#######################################################################################

## recupérer le nom du binaire en argument CLI 
if [ -z "$1" ] || [ -z "$2" ]; then 
	echo "ERROR : Usage : $0 <profiling_mode> <kernel_binary>"
	echo "Modes disponibles : cqa, lprof, oneview"
#	exit 1
fi
MODE="$1"
KERNEL="$2"



# commandes maqao
function maqao_cqa {
	echo "Profiling with MAQAO CQA on binary : ${KERNEL}."
	maqao cqa ${KERNEL} fct-loops=main conf=all --output-format=html
}

function maqao_lprof {
        echo "Profiling with MAQAO LPROF on binary : ${KERNEL}."
        maqao lprof ${KERNEL}
}

function maqao_oneview {
        echo "Profiling with MAQAO ONEVIEW on binary : ${KERNEL}."
        maqao oneview -R1 --output-format=html -xp=maqao_exp -- ${KERNEL} 
}


case "$MODE" in 
	cqa)
		maqao_cqa
		;;
	lprof)
		maqao_lprof
		;;
	oneview)
		maqao_oneview
		;;
	*)
		echo "Mode invalide : $MODE"
        echo "Modes disponibles : cqa, lprof, oneview"
#		exit 1
		;;
esac



