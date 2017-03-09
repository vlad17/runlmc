#!/bin/bash
# Shared argument-parsing and common control-flow file for benchmark
# ./run.sh files
#
# First argument should be a string
# that can be printf-d into print help.
# The rest of the arguments should be the same as what was passed to run.sh.

# Return code indicates what happened.
# 0 - normal execution
# 1 - validation (small) benchmark
# 2 - printed help
# 3 - error

HELP_STR="$1"
shift

USAGE="Usage: ./run.sh [--help|--validate]"

function print_help() {
    echo $USAGE
    printf "$HELP_STR"
    echo
    echo "Flags"
    echo "    --validate Run a small case to verify configuration."
    echo "    --help Print this help message."
}


if [[ $# -gt 1 ]]; then
    echo $USAGE >/dev/stderr
    exit 3
fi

# Return code indicates what happened.
# 0 - normal execution
# 1 - validation (small) benchmark
# 2 - printed help
# 3 - error

if [[ $# -eq 1 ]]; then
    case $1 in
        "--help")
            print_help
            exit 2
            ;;
        "--validate")
            exit 1
            ;;
        *)
            echo $USAGE >/dev/stderr
            exit 3
            ;;
    esac
else
    exit 0
fi
