SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


CURRENT_PATH=$(pwd)

kernel_list=(
#	"demos/FiniteVolume/finite-volume-level-set"
#	"demos/FiniteVolume/finite-volume-advection-2d"
	"demos/tutorial/tutorial-graduation-case-2"
	)


# !!! For now, dont work 
arg_list=(
#	"--max-level 6"
#	"--max-level 4"
	"--max-level 8"
)


for i in "${!kernel_list[@]}"; do 
	binary="${kernel_list[$i]}"
	args="${arg_list[$i]}"
	echo "Run $binary $args"
	. "${SCRIPT_DIR}/run.sh" "${CURRENT_PATH}/${binary}" # $args
done
