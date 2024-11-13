case=B;
M=3;
solver=gmres;
for precision in 30 40; do
python3 local_mtf.py --M $M --case $case --precision $precision --solver $solver;
#python3 stf.py --M $M --case $case --precision $precision --solver $solver;
done
wait;