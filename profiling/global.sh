

kernel_list=(
#	"demos/FiniteVolume/finite-volume-level-set"
	"demos/FiniteVolume/finite-volume-advection-2d"
	"demos/tutorial/tutorial-graduation-case-2"
	)

arg_list=(
#	"--max-level 6"
	"--max-level 10"
	"--max-level 14"
)


# should verify if same size


for i in "${!kernel_list[@]}"; do 
	binary="${kernel_list[$i]}"
	args="${arg_list[$i]}"
	echo "Run $binary $args"
	. ../profiling/run.sh ./$binary # $args
done
