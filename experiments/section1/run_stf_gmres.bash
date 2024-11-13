case=B;
M=2;
solver=gmres;
for precision in 1 2 5 10 20 30 40 50 60; do
#python3 main.py --M $M --case $case --precision $precision;
python3 stf.py --M $M --case $case --precision $precision --solver $solver;
done
wait;