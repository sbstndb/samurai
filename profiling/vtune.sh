#######################################################################################
## Objectif : 		pouvoir facilement lancer les commandes VTUNE classiques
## Outil VTUNE : 	permet d'obtenir des mesures de performances (call tree, etc)
## Fonctionnement : 	. vtune.sh <mode> <binary>
## Exemple : 		. profilers/vtune.sh hotspot ./build/demos/tutorial/reconstruction
#######################################################################################

## recupérer le nom du binaire en argument CLI 
if [ -z "$1" ] || [ -z "$2" ]; then 
	echo "ERROR : Usage : $0 <profiling_mode> <kernel_binary>"
	echo "Modes disponibles : hotspots"
#	exit 1
fi
MODE="$1"
KERNEL="$2"



function vtune_hotspots {
        echo "Profiling with VTUNE HOTSPOTS on binary : ${KERNEL}."
        vtune -collect hotspots ${KERNEL} 
}


case "$MODE" in 
	hotspots)
		vtune_hotspots
		;;
	*)
		echo "Mode invalide : $MODE"
        echo "Modes disponibles : hotspots"
#		exit 1
		;;
esac



