case=A;
M=2;
solver=direct;
for precision in 1 2 5 10 20 30 40; do
#python3 main.py --M $M --case $case --precision $precision;
python3 stf.py --M $M --case $case --precision $precision --solver $solver;
done
wait;