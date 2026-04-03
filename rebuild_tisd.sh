#!/bin/bash

# --- TISD Automated Pipeline Rebuild v3.0 ---
# Function to draw the progress bar
draw_progress_bar() {
    local percentage=$1
    local width=40
    local completed=$((percentage * width / 100))
    local remaining=$((width - completed))
    
    printf "\r["
    for ((i=0; i<completed; i++)); do printf "#"; done
    for ((i=0; i<remaining; i++)); do printf "-"; done
    printf "] %d%%" "$percentage"
}

echo "🎒 TISD: Initializing M4 Data Pipeline..."
start_total=$(date +%s)

# Helper function for timestamps
timestamp() {
  date +"[%H:%M:%S]"
}

# START: 0%
draw_progress_bar 0
echo -e "\n$(timestamp) 🛠️  Cleaning up..."
rm -rf notebooks/*_executed.ipynb

# STAGE 1: 0% -> 50%
echo "------------------------------------------------"
echo "📄 [STEP 1/2] PDF Extraction & Chunking"
echo "Starting 01_data_prep.ipynb..."
echo "------------------------------------------------"

set -e
# Executing 01. Progress jumps to 25% while running, 50% when done.
draw_progress_bar 10
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_prep.ipynb > /dev/null 2>&1
draw_progress_bar 50

echo -e "\n$(timestamp) ✅ Step 1 Complete."

# STAGE 2: 50% -> 100%
echo "------------------------------------------------"
echo "🧠 [STEP 2/2] Embedding & Vector Indexing"
echo "Starting 02_embeddings.ipynb..."
echo "------------------------------------------------"

# Progress jumps to 75% while running, 100% when done.
draw_progress_bar 75
jupyter nbconvert --to notebook --execute --inplace notebooks/02_embeddings.ipynb > /dev/null 2>&1
draw_progress_bar 100

echo -e "\n$(timestamp) ✅ Step 2 Complete."
set +e

end_total=$(date +%s)
runtime=$((end_total-start_total))

echo -e "\n================================================"
echo "📊 PIPELINE SUMMARY"
echo "Status: SUCCESS"
echo "Duration: $runtime seconds"
echo "Next Step: Run '03b_mlx_engine.ipynb' to start Tara."
echo "================================================"