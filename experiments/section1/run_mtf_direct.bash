case=B;
solver=direct;
M=3;
for precision in 1 2 5 10 20 30 40; do
python3 local_mtf.py --M $M --case $case --precision $precision --solver $solver;
done
wait;