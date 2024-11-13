# Section 1

We study and verify the convergence of the local MTF for a unit sphere, comparing the STF to local MTF for $M=2$ and $M=3$ (i.e. scattering by two half-spheres).

To plot the solution for cases A and B and `precision = 10`, use the Notebook: `mtf.ipynb`.

To generate all the simulations for convergence:

For STF(2), case A:

```
bash run_stf_direct.bash
```

Edit `run_stf_direct.bash` for `case=B`.

For MTF(2), case A:

```
bash run_mtf_direct.bash # Set case A and B
```

Edit `run_mtf_direct.bash` for `case=B`.

For case B and `precision in [30,40]`, run:

```
bash run_mtf_gmres.bash # Set case A and B
```

Next, plot the figures through: `plot_results.ipynb`.