# Structure-preserving PHS RBF-FD methods for Schrödinger equations

Minimal research-code and numerical-data archive accompanying the manuscript **Structure-Preserving PHS RBF-FD Methods for Schrödinger Equations**.

## Included
- `free_schrodinger/src/`: C++17 free-equation and higher-grid implementations.
- `free_schrodinger/scripts/`: numerical checks and plotting scripts.
- `free_schrodinger/data/`: selected underlying numerical CSVs for modal convergence, stencil sensitivity, and basis screening.
- `variable_potential/`: variable-potential C++ implementation and selected six-grid results.
- `certificates/`: coefficient certificate scripts and recorded outputs.

## Build and run (MSYS2 UCRT64 / CMake + Ninja)

```bash
cmake -S free_schrodinger -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/schrodinger_experiments free_schrodinger/data
```

C++ requires Boost.Multiprecision headers. Plotting/check scripts require Python with numpy, scipy, pandas and matplotlib. For example:

```bash
python free_schrodinger/scripts/plot_all.py --data free_schrodinger/data --figures figures
```

## Scope and limitations
This is a **minimal selected-code/data release**, not a complete archival reproduction of every manuscript table or figure. Generated figures, full matrix datasets, complete reference-refinement data, and the manuscript itself are omitted to keep the repository small. Running the C++ executable can generate additional CSVs. The two certificate log files are recorded results, not evidence of independent reruns in this release. The numerical values in the variable-potential CSV should be checked against the final manuscript before a formal archival release.

No license is included; add one only after both authors agree on reuse terms.
