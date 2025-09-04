#!/usr/bin/env bash

set -euo pipefail

# ----------------------------------------------------------------------------
# run_mpi_commits.sh
#
# Pour chaque commit (en partant de HEAD), crée un nouveau dossier, extrait
# les sources de ce commit, compile les 3 démos FiniteVolume avec MPI, et lance
# les calculs MPI demandés. Les sorties sont loggées dans chaque dossier run.
#
# Compilation CMake:
#   -DWITH_MPI=ON
#   -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O3 -g"
#
# Exécutions:
#   1) mpirun -np 2 ./demos/FiniteVolume/finite-volume-burgers --max-level 10 --min-level 6 --Tf 1.5 --nfiles 1000
#   2) mpirun -np 8 ./demos/FiniteVolume/finite-volume-burgers --max-level 10 --min-level 6 --Tf 1.5 --nfiles 10000
#   3) mpirun -np 8 ./demos/FiniteVolume/finite-volume-linear-convection --max-level 7 --min-level 5 --Tf 0.8 --nfiles 1000
#   4) mpirun -np 8 ./demos/FiniteVolume/finite-volume-advection-2d --max-level 9 --min-level 6 --Tf 0.8 --nfiles 1000
#
# Remarques:
# - Les démos sont définies sous demos/FiniteVolume/CMakeLists.txt.
# - Ce script évite BUILD_DEMOS=ON pour ne pas construire d'autres sous-projets
#   potentiellement non disponibles (p4est/pablo/…). Il compile uniquement les
#   cibles nécessaires via --target.
# - Les sources pour chaque commit sont extraites via `git archive` pour ne pas
#   modifier l'état du dépôt courant, et garantir « un dossier par commit ».
# - Ce script n'installe pas de dépendances. Assurez-vous que vos dépendances
#   (HDF5 parallèle, Boost.MPI, HighFive, xtensor, fmt, pugixml, etc.) sont
#   disponibles dans l'environnement.
# ----------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage: scripts/run_mpi_commits.sh [options]

Options:
  --limit N            Nombre de commits à traiter (défaut: 5)
  --start-ref REF      Référence de départ (défaut: HEAD)
  --jobs N             Nombre de jobs pour la compilation (défaut: nproc)
  --out DIR            Dossier racine des runs (défaut: mpi_runs)
  --timeout SECS       Timeout en secondes par exécution (0 = illimité, défaut: 0)
  --concurrent-commits N  Nombre de commits en parallèle (défaut: 1)
  --help                  Affiche cette aide

Exemple:
  scripts/run_mpi_commits.sh --limit 10 --jobs 8 --out my_runs
EOF
}

log()   { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# Defaults
LIMIT=5
START_REF="HEAD"
JOBS=$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)
OUT_ROOT="mpi_runs"
TIMEOUT_SECS=0
CONCURRENT_COMMITS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)      LIMIT="${2:-}"; shift 2 ;;
    --start-ref)  START_REF="${2:-}"; shift 2 ;;
    --jobs)       JOBS="${2:-}"; shift 2 ;;
    --out)        OUT_ROOT="${2:-}"; shift 2 ;;
    --timeout)    TIMEOUT_SECS="${2:-}"; shift 2 ;;
    --concurrent-commits) CONCURRENT_COMMITS="${2:-}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            warn "Option inconnue: $1"; usage; exit 2 ;;
  esac
done

# Tools
GIT=${GIT:-git}
CMAKE=${CMAKE:-cmake}
MPI_RUN=${MPI_RUN:-mpirun}

command -v "$GIT"   >/dev/null 2>&1 || error "git introuvable"
command -v "$CMAKE" >/dev/null 2>&1 || error "cmake introuvable"

if ! command -v "$MPI_RUN" >/dev/null 2>&1; then
  if command -v mpiexec >/dev/null 2>&1; then
    MPI_RUN=mpiexec
  else
    error "mpirun/mpiexec introuvable dans le PATH"
  fi
fi

# Check git repo
"$GIT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || error "Ce script doit être lancé dans un dépôt git."
REPO_ROOT=$("$GIT" rev-parse --show-toplevel)

# Resolve commits list
log "Récupération des $LIMIT commits depuis $START_REF (first-parent)…"
mapfile -t COMMITS < <( "$GIT" rev-list --first-parent --max-count="$LIMIT" "$START_REF" )
if [[ ${#COMMITS[@]} -eq 0 ]]; then
  error "Aucun commit trouvé depuis $START_REF"
fi

mkdir -p "$OUT_ROOT"

# Summary
SUMMARY_FILE="$OUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Run summary (start: $(date))" > "$SUMMARY_FILE"
echo "Start ref: $START_REF"           >> "$SUMMARY_FILE"
echo "Commits: ${#COMMITS[@]}"         >> "$SUMMARY_FILE"
echo "Concurrent commits: $CONCURRENT_COMMITS" >> "$SUMMARY_FILE"
echo ""                                 >> "$SUMMARY_FILE"

sanitize() { echo "$1" | tr ' /:\\' '____' | tr -cd '[:alnum:]_.-' | cut -c1-60; }

run_with_timeout() {
  # $1 = timeout secs or 0, rest = command
  local t="$1"; shift
  if [[ "$t" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout "$t" "$@"
  else
    "$@"
  fi
}

# Thread-safe append to summary using flock (if available)
append_summary() {
  local line="$1"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -w 10 9 || true
      echo "$line" >&9
    ) 9>>"$SUMMARY_FILE"
  else
    # Fallback without flock (possible interleaving)
    echo "$line" >> "$SUMMARY_FILE"
  fi
}

build_and_run_for_commit() {
  local commit_sha="$1"
  local idx="$2"

  local short_sha; short_sha=$("$GIT" rev-parse --short "$commit_sha")
  local title; title=$("$GIT" log -1 --pretty=format:%s "$commit_sha")
  local stitle; stitle=$(sanitize "$title")

  local run_dir="$OUT_ROOT/${idx}_$short_sha"; mkdir -p "$run_dir"; run_dir=$(cd "$run_dir" && pwd)
  local src_dir="$run_dir/src";            mkdir -p "$src_dir"
  local build_dir="$run_dir/build";        mkdir -p "$build_dir"
  local logs_dir="$run_dir/logs";          mkdir -p "$logs_dir"

  log "[$idx/$LIMIT] Commit $short_sha: $title"
  append_summary "[$(date)] $short_sha $title"

  # Extract sources for this commit without touching current worktree
  log "Extraction des sources du commit $short_sha → $src_dir"
  "$GIT" -C "$REPO_ROOT" archive "$commit_sha" | tar -x -C "$src_dir" || {
    warn "Extraction échouée pour $short_sha"; echo "  extract: FAIL" >> "$SUMMARY_FILE"; return 0; }

  # Configure
  log "Configuration CMake (WITH_MPI=ON, flags optimisés)"
  local cxx_flags="-mtune=native -march=native -O3 -g"
  # Configure and build each target in its own build dir to allow true parallelization
  local build_dir_burg="$run_dir/build_burgers"; mkdir -p "$build_dir_burg"; build_dir_burg=$(cd "$build_dir_burg" && pwd)
  local build_dir_lin="$run_dir/build_linear_conv"; mkdir -p "$build_dir_lin"; build_dir_lin=$(cd "$build_dir_lin" && pwd)
  local build_dir_adv="$run_dir/build_advection_2d"; mkdir -p "$build_dir_adv"; build_dir_adv=$(cd "$build_dir_adv" && pwd)

  log "Configuration CMake pour 3 cibles (parallèle)"

  local -a pids=()
  local jobs_per_target=1
  if [[ "$JOBS" -gt 3 ]]; then
    jobs_per_target=$(( (JOBS + 2) / 3 ))
  fi

  configure_and_build() {
    # $1: target name, $2: build dir
    local target="$1"; local bdir="$2"
    local tlog="$logs_dir/${target}"
    mkdir -p "$logs_dir"
    if ! "$CMAKE" -S "$src_dir" -B "$bdir" \
        -DWITH_MPI=ON \
        -DCMAKE_CXX_FLAGS="$cxx_flags" \
        -DCMAKE_BUILD_TYPE=Release \
        >"${tlog}_configure.out" 2>"${tlog}_configure.err"; then
      echo "CONFIGURE_FAIL" >"${tlog}_status.txt"
      return 0
    fi
    if ! "$CMAKE" --build "$bdir" -j"$jobs_per_target" --target "$target" \
        >"${tlog}_build.out" 2>"${tlog}_build.err"; then
      echo "BUILD_FAIL" >"${tlog}_status.txt"
      return 0
    fi
    echo "OK" >"${tlog}_status.txt"
  }

  configure_and_build finite-volume-burgers "$build_dir_burg" & pids+=("$!")
  configure_and_build finite-volume-linear-convection "$build_dir_lin" & pids+=("$!")
  configure_and_build finite-volume-advection-2d "$build_dir_adv" & pids+=("$!")

  # Wait for all three
  for pid in "${pids[@]}"; do wait "$pid"; done

  # Check statuses
  local st_burg st_lin st_adv
  st_burg=$(cat "$logs_dir/finite-volume-burgers_status.txt" 2>/dev/null || echo FAIL)
  st_lin=$(cat "$logs_dir/finite-volume-linear-convection_status.txt" 2>/dev/null || echo FAIL)
  st_adv=$(cat "$logs_dir/finite-volume-advection-2d_status.txt" 2>/dev/null || echo FAIL)

  if [[ "$st_burg" != OK || "$st_lin" != OK || "$st_adv" != OK ]]; then
    warn "Compilation échouée pour $short_sha (burgers=$st_burg, lin=$st_lin, adv2d=$st_adv)"
    append_summary "  build: FAIL (burgers=$st_burg, lin=$st_lin, adv2d=$st_adv)"
    append_summary ""
    return 0
  fi

  # Executables paths (each in its own build)
  local exe_burgers="$build_dir_burg/demos/FiniteVolume/finite-volume-burgers"
  local exe_linconv="$build_dir_lin/demos/FiniteVolume/finite-volume-linear-convection"
  local exe_adv2d="$build_dir_adv/demos/FiniteVolume/finite-volume-advection-2d"

  # Check existence
  for exe in "$exe_burgers" "$exe_linconv" "$exe_adv2d"; do
    if [[ ! -x "$exe" ]]; then
      warn "Binaire manquant: $exe"
    fi
  done

  # Runs
  log "Exécutions MPI sur $short_sha (logs sous $logs_dir, outputs isolés)"

  # Ensure per-run output directories (isolated per commit)
  local out_root="$run_dir/outputs"; mkdir -p "$out_root"
  local out_burg_np2="$out_root/burgers_np2_nfiles1000"; mkdir -p "$out_burg_np2"
  local out_burg_np8="$out_root/burgers_np8_nfiles10000"; mkdir -p "$out_burg_np8"
  local out_lin_np8="$out_root/linear_convection_np8_nfiles1000"; mkdir -p "$out_lin_np8"
  local out_adv_np8="$out_root/advection_2d_np8_nfiles1000"; mkdir -p "$out_adv_np8"

  run_in_dir() {
    # $1: workdir, rest: command
    local workdir="$1"; shift
    ( cd "$workdir" && run_with_timeout "$TIMEOUT_SECS" "$@" )
  }

  local ok_build=1
  local status1 status2 status3 status4

  # 1) Burgers np=2, nfiles=1000
  if [[ -x "$exe_burgers" ]]; then
    log "Run 1/4: burgers np=2 nfiles=1000"
    if run_in_dir "$out_burg_np2" "$MPI_RUN" -np 2 "$exe_burgers" --max-level 10 --min-level 6 --Tf 1.5 --nfiles 1000 \
        >"$logs_dir"/run_burgers_np2_nfiles1000.out 2>"$logs_dir"/run_burgers_np2_nfiles1000.err; then
      status1=OK
    else
      status1=FAIL; ok_build=0
    fi
  else
    status1=SKIP
  fi

  # 2) Burgers np=8, nfiles=10000
  if [[ -x "$exe_burgers" ]]; then
    log "Run 2/4: burgers np=8 nfiles=10000"
    if run_in_dir "$out_burg_np8" "$MPI_RUN" -np 8 "$exe_burgers" --max-level 10 --min-level 6 --Tf 1.5 --nfiles 10000 \
        >"$logs_dir"/run_burgers_np8_nfiles10000.out 2>"$logs_dir"/run_burgers_np8_nfiles10000.err; then
      status2=OK
    else
      status2=FAIL; ok_build=0
    fi
  else
    status2=SKIP
  fi

  # 3) Linear convection np=8, nfiles=1000
  if [[ -x "$exe_linconv" ]]; then
    log "Run 3/4: linear-convection np=8 nfiles=1000"
    if run_in_dir "$out_lin_np8" "$MPI_RUN" -np 8 "$exe_linconv" --max-level 7 --min-level 5 --Tf 0.8 --nfiles 1000 \
        >"$logs_dir"/run_linear_convection_np8_nfiles1000.out 2>"$logs_dir"/run_linear_convection_np8_nfiles1000.err; then
      status3=OK
    else
      status3=FAIL; ok_build=0
    fi
  else
    status3=SKIP
  fi

  # 4) Advection 2D np=8, nfiles=1000
  if [[ -x "$exe_adv2d" ]]; then
    log "Run 4/4: advection-2d np=8 nfiles=1000"
    if run_in_dir "$out_adv_np8" "$MPI_RUN" -np 8 "$exe_adv2d" --max-level 9 --min-level 6 --Tf 0.8 --nfiles 1000 \
        >"$logs_dir"/run_advection_2d_np8_nfiles1000.out 2>"$logs_dir"/run_advection_2d_np8_nfiles1000.err; then
      status4=OK
    else
      status4=FAIL; ok_build=0
    fi
  else
    status4=SKIP
  fi

  append_summary "  build: OK"
  append_summary "  runs : burgers(n2,n1000)=${status1:-SKIP}, burgers(n8,n10000)=${status2:-SKIP}, linconv(n8,n1000)=${status3:-SKIP}, adv2d(n8,n1000)=${status4:-SKIP}"
  append_summary ""
}

log "Dossier cible des runs: $OUT_ROOT"
log "Nombre de commits: ${#COMMITS[@]} (limit=$LIMIT)"
log "Commits en parallèle: $CONCURRENT_COMMITS"

trap 'wait' EXIT

idx=0
running=0
for c in "${COMMITS[@]}"; do
  idx=$((idx+1))
  build_and_run_for_commit "$c" "$idx" &
  running=$((running+1))
  # Limite de parallélisme
  while [[ "$running" -ge "$CONCURRENT_COMMITS" ]]; do
    if wait -n 2>/dev/null; then
      running=$((running-1))
    else
      # wait -n indisponible (bash < 5), fallback: attendre tous
      wait
      running=0
    fi
  done
done

# Attendre les derniers jobs
wait || true

log "Terminé. Résumé: $SUMMARY_FILE"
exit 0
