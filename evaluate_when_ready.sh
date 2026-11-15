#!/bin/bash
# Auto-evaluate stoppage cues and ball recovery when processes complete

set -e

echo "Waiting for stoppage cue and ball recovery processes to complete..."

# Wait for output files to exist
while [ ! -f "submissions/stoppage_cues_public.csv" ] || [ ! -f "submissions/improved_with_ball_recovery.csv" ]; do
    echo "Waiting... ($(date '+%H:%M:%S'))"
    sleep 30
done

echo "Files detected! Waiting 10 seconds to ensure writing is complete..."
sleep 10

echo "=== Stoppage Cues ==="
cat submissions/stoppage_cues_public.csv
echo ""
echo "Stoppage cues detected: $(tail -n +2 submissions/stoppage_cues_public.csv | wc -l)"
echo ""

echo "=== Evaluating Ball Recovery + Improved Dead Ball ==="
cp submissions/improved_with_ball_recovery.csv evaluation/results/
.venv/bin/python evaluation/run_evaluation.py --no-detail | tee evaluation_results.txt

echo ""
echo "=== Results Summary ==="
grep -A 15 "^rank" evaluation_results.txt | head -20

echo ""
echo "=== Next Steps ==="
echo "1. Check if ball recovery improved results"
echo "2. If successful, integrate stoppage cues:"
echo "   .venv/bin/python tactical_baseline.py \\"
echo "     --data-dir data \\"
echo "     --ball-detector models/yolo-sn-ball-opt.pt \\"
echo "     --stoppage-cues submissions/stoppage_cues_public.csv \\"
echo "     --detector-range 11185 18685 \\"
echo "     --output submissions/full_integration.csv"
echo ""
echo "3. Evaluate combined approach"
echo "4. If improvements confirmed, commit changes"

echo ""
echo "Evaluation complete at $(date)"
