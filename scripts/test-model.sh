#!/bin/bash

# --- Parallel Request Testing Script ---
#
# This script sends simultaneous POST requests to a specified URL.
# Each request is given a unique payload to prevent caching.
#
# Usage:
# ./parallel_request_tester.sh -r 20
#   -r : (Optional) Number of requests to run. Defaults to 10.
#
# -----------------------------------------------------------------

# --- Configuration ---

# Default number of requests
REQUEST_COUNT=10

# The URL to test.
# IMPORTANT: This is set to a local placeholder.
# Change this to your own local development server's endpoint.
URL="https://granite-inference-demo.apps.prime.pitt.ca/v1/chat/completions"

# --- Usage Function ---
usage() {
  echo "Usage: $0 [-r <number_of_requests>]"
  echo "  -r : Number of simultaneous requests to run (default: 10)"
  exit 1
}

# --- Parse Command-Line Options ---
while getopts "r:" opt; do
  case $opt in
    r)
      # Basic validation to ensure it's a positive integer
      if ! [[ "$OPTARG" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Number of requests must be a positive integer." >&2
        usage
      fi
      REQUEST_COUNT=$OPTARG
      ;;
    \?)
      usage
      ;;
  esac
done


# --- Function to run a single request ---
# This function will be run in the background
# $1: The request number (i)
run_request() {
  local i=$1

  # --- Generate a unique JSON payload for each request ---
  # We embed the request number '$i' in the content to make it unique.
  # Note: We use "EOF" (no quotes) to allow $i to be expanded.
  local JSON_PAYLOAD
  JSON_PAYLOAD=$(cat <<EOF
{
  "model": "granite-3.1-8b-instruct-quantized.w4a16",
  "messages": [{"role": "user", "content": "What is AI? Write me a long story about it (Request $i)"}],
  "temperature": 0.8
}
EOF
)

  echo "Starting request $i..."

  # Get start time in seconds
  local start_time=$(date +%s)

  # Run the curl command, saving output and errors to files
  curl -k -s -X POST "$URL" \
       -H "Content-Type: application/json" \
       -d "$JSON_PAYLOAD" \
       -o "output_$i.txt" 2> "error_$i.txt"

  # Get end time and calculate duration
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))

  # Removed the "waiting" line as requested
  echo "  - request $i: received response in ${duration}s"
}

# Export the function and URL variable so they are available
# to the backgrounded (sub-shell) processes.
export -f run_request
export URL
# We no longer export JSON_PAYLOAD as it's created locally in the function


echo "Starting $REQUEST_COUNT simultaneous requests to $URL..."
echo "------------------------------------------------"

# Loop $REQUEST_COUNT times to start background processes
for i in $(seq 1 $REQUEST_COUNT)
do
  # Run the function in the background
  run_request "$i" &
done

# 'wait' command blocks the script until all background
# jobs (the function calls) have finished.
echo "Waiting for all requests to complete..."
wait

echo "------------------------------------------------"
echo "All $REQUEST_COUNT requests complete."
echo "Output saved to files named output_1.txt through output_${REQUEST_COUNT}.txt"
echo "Errors (if any) saved to error_1.txt through error_${REQUEST_COUNT}.txt"
echo ""
echo "Example output from request 1 (output_1.txt):"
if [ -f "output_1.txt" ]; then
  # Check if the output file is not empty
  if [ -s "output_1.txt" ]; then
    cat "output_1.txt"
  else
    echo "output_1.txt was created but is empty (check error log)."
  fi
else
  echo "output_1.txt not found (request likely failed)."
fi
echo "" # Newline for clarity
echo "Example error from request 1 (error_1.txt):"
if [ -f "error_1.txt" ] && [ -s "error_1.txt" ]; then
  # Check if error file exists and is not empty
  cat "error_1.txt"
else
  echo "No error logged for request 1."
fi
echo "" # Newline for clarity
