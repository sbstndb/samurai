import sys
import subprocess
import re
import statistics
from collections import defaultdict

# Configuration
MIN_RUNS = 5  # Minimum number of runs before checking for stability
MAX_RUNS = 100  # Maximum number of runs to prevent infinite loops
STABILITY_THRESHOLD = 0.02  # 5% coefficient of variation

def parse_timers(output):
    """Parses the timer output of the samurai simulation."""
    metrics = {}
    # Regex to capture the metric name and its min, max, and average times.
    # It handles lines with optional MPI rank output like:
    # '         data saving             0.12967 [   2 ]             0.14172 [   3 ]             0.13461             0.00356                   2'
    # It captures: 1. Name, 2. Min time, 3. Max time, 4. Ave time
    pattern = re.compile(r"^\s+([\w\s]+?)\s{2,}([\d.]+)(?:\s+\[\s+\d+\s+\])?\s+([\d.]+)(?:\s+\[\s+\d+\s+\])?\s+([\d.]+)\s+.*$")

    lines = output.split('\n')
    for line in lines:
        match = pattern.search(line)
        if match:
            # Clean up the metric name and convert time to float
            metric_name = match.group(1).strip()
            min_time = float(match.group(2))
            max_time = float(match.group(3))
            avg_time = float(match.group(4))
            metrics[metric_name] = {
                "min": min_time,
                "max": max_time,
                "ave": avg_time,
            }
    return metrics

def main():
    """Main function to run the stabilization script."""
    if len(sys.argv) < 2:
        print("Usage: python stabilizer.py <command...>")
        print("Example: python stabilizer.py mpirun -np 4 demos/FiniteVolume/finite-volume-advection-2d --nfiles 1 --timers")
        sys.exit(1)

    command = sys.argv[1:]
    
    print(f"Running command: {' '.join(command)}")
    
    # Dictionary to store the list of results for each metric
    all_metrics = defaultdict(lambda: defaultdict(list))
    
    for i in range(MAX_RUNS):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            current_metrics = parse_timers(result.stdout)
            if not current_metrics:
                sys.stdout.write("\r")
                print("Error: Could not parse any metrics from the output.")
                print("--- STDOUT ---")
                print(result.stdout)
                print("--- STDERR ---")
                print(result.stderr)
                sys.exit(1)

            for name, values in current_metrics.items():
                for time_type, value in values.items():
                    all_metrics[name][time_type].append(value)

        except subprocess.CalledProcessError as e:
            sys.stdout.write("\r")
            print(f"Command failed with exit code {e.returncode}")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            sys.exit(1)
        
        # --- Stability Check ---
        if i + 1 >= MIN_RUNS:
            is_stable = True
            max_cv = 0
            unstable_count = 0
            total_metrics_count = 0

            for name, values_dict in all_metrics.items():
                for time_type, values in values_dict.items():
                    if len(values) >= 2:
                        total_metrics_count += 1
                        mean = statistics.mean(values)
                        stdev = statistics.stdev(values)
                        cv = stdev / mean if mean != 0 else 0
                        if cv > max_cv:
                            max_cv = cv
                        
                        if cv > STABILITY_THRESHOLD:
                            is_stable = False
                            unstable_count += 1
            
            if total_metrics_count > 0:
                stable_metrics_count = total_metrics_count - unstable_count
                status_msg = f"Iteration {i + 1}/{MAX_RUNS}: {stable_metrics_count}/{total_metrics_count} stable metrics (Max CV: {max_cv:.4f})"
                # Pad with spaces to clear the line
                sys.stdout.write(f"\r{status_msg.ljust(80)}")
                sys.stdout.flush()

            # Display current statistics for each metric
            if i + 1 >= MIN_RUNS:
                sys.stdout.write("\n")
                for name, values_dict in all_metrics.items():
                    sys.stdout.write(f"  {name}:\n")
                    for time_type, values in values_dict.items():
                        if values:
                            median_value = statistics.median(values)
                            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                            sys.stdout.write(f"    {time_type}: {median_value:.6e}s ± {stdev:.6e}s\n")
                sys.stdout.write("\033[F" * (1 + sum(1 + len(v) for v in all_metrics.values())))  # Move cursor up to overwrite next time
                sys.stdout.flush()

            if is_stable and total_metrics_count > 0:
                sys.stdout.write("\n\nMetrics have stabilized.\n")
                break
        else:
            status_msg = f"Iteration {i + 1}/{MAX_RUNS}: Collecting data..."
            sys.stdout.write(f"\r{status_msg.ljust(80)}")
            sys.stdout.flush()
    else: # This 'else' is for the 'for' loop, executed if 'break' is not hit
        print(f"\n\nMax runs ({MAX_RUNS}) reached. Metrics may not be fully stable.")

    # --- Final Results ---
    print("\n--- Final Results (median ± stdev) ---")
    final_results = {}
    for name, values_dict in all_metrics.items():
        final_results[name] = {}
        print(f"  {name}:")
        for time_type, values in values_dict.items():
            if values:
                median_value = statistics.median(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0.0
                final_results[name][time_type] = {"median": median_value, "stdev": stdev}
                print(f"    {time_type}: {median_value:.6e}s ± {stdev:.6e}s")

if __name__ == "__main__":
    main()
