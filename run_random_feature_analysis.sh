# Script to run all random feature experiments
out_dir="results/random_feature_analysis"
mkdir -p "$out_dir"

for fp_type in "count" "binary" ; do
    if [[ $fp_type == "count" ]]; then
        fp_arg=""
    elif [[ $fp_type == "binary" ]]; then
        fp_arg="--binary_fps"
    else
        echo "Invalid fingerprint type"
        exit 1
    fi

    # T_MM
    echo "Running T_MM random feature analysis for $fp_type fingerprints"
    PYTHONPATH=.:"$PYTHONPATH" python experiment_scripts/tmm_random_feature_analysis.py \
        --output_json "$out_dir"/"tmm_${fp_type}_fp.json" \
        $fp_arg

    # T_DP
    echo "Running T_DP random feature analysis for $fp_type fingerprints"
    PYTHONPATH=.:"$PYTHONPATH" python experiment_scripts/tdp_random_feature_analysis.py \
        --output_json "$out_dir"/"tdp_${fp_type}_fp.json" \
        $fp_arg

done
