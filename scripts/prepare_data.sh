#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_data.sh [-h|--help]

Download and prepare WiLI-2018 data

Optional arguments:
  -h, --help    Show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# download and prepare Facebook multi-class NLU data set
prepare_data() {
  local directory="./data/"
  wget -N -P "$directory" \
    "https://zenodo.org/record/841984/files/wili-2018.zip"
  unzip "$directory/wili-2018.zip" -d "$directory/wili-2018/"
}

# execute all functions
check_help "$@"
prepare_data
