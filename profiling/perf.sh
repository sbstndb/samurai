#######################################################################################
## Objectif : 		pouvoir facilement lancer les commandes perf stat classiques
## Outil perf : 	permet d'extraire des valeurs cpu de performance
## Fonctionnement : 	. perf.sh <mode> <binary>
## Exemple : 		. profilers/perf.sh all ./build/demos/tutorial/reconstruction
#######################################################################################

## recupérer le nom du binaire en argument CLI 
if [ -z "$1" ] || [ -z "$2" ]; then 
	echo "ERROR : Usage : $0 <profiling_mode> <kernel_binary>"
	echo "Modes disponibles : hw, sw, cache, syscall, all "
#	exit 1
fi
MODE="$1"
KERNEL="$2"


# alias pour simplifier 

PERF_STAT_HW="branches,branch-misses,bus-cycles,cache-misses,cache-references,cpu-cycles,instructions,ref-cycles"
PERF_STAT_SW="alignment-faults,context-switches,cpu-clock,cpu-migrations" 
PERF_STAT_CACHE_L1="L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores"
PERF_STAT_CACHE_LLC="LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
PERF_STAT_CACHE_BRANCH="branch-load,branch-load-misses"
PERF_STAT_CACHE="${PERF_STAT_CACHE_L1},${PERF_STAT_CACHE_LLC},${PERF_STAT_CACHE_BRANCH}"
PERF_STAT_SYSCALL="raw_syscalls:sys_enter"


# commandes perf
function perf_hw {
	echo "Profiling HW with perf stat on binary : ${KERNEL}."
	sudo perf stat -e ${PERF_STAT_HW} ${KERNEL}
}

function perf_sw {
        echo "Profiling SW with perf stat on binary : ${KERNEL}."
        sudo perf stat -e ${PERF_STAT_SW} ${KERNEL}
}

function perf_cache {
        echo "Profiling CACHE with perf stat on binary : ${KERNEL}."
        sudo perf stat -e ${PERF_STAT_CACHE} ${KERNEL}
}

function perf_syscall {
        echo "Profiling SYSCALL with perf stat on binary : ${KERNEL}."
        sudo perf stat -e ${PERF_STAT_SYSCALL} ${KERNEL}
}

function perf {
        echo "Profiling with perf stat on binary : ${KERNEL}."
        sudo perf stat -e "${PERF_STAT_HW},${PERF_STAT_SW},${PERF_STAT_CACHE},${PERF_STAT_SYSCALL}" ${KERNEL}
}


case "$MODE" in 
	hw)
		perf_hw
		;;
	sw)
		perf_sw
		;;
	cache)
		perf_cache
		;;
	syscall)
		perf_syscall
		;;
	all)
		perf
		;;
	*)
		echo "Mode invalide : $MODE"
		echo "Modes disponibles = hw, sw, cache, syscall, all"
#		exit 1
		;;
esac



