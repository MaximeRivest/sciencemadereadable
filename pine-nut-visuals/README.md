# Pine-nut study: visual companion

Open `index.html`. Everything runs locally in one file; no libraries, accounts, or data services are needed. Page colours follow Chattering's theme variables. Patterns, labels and symbols also work without colour.

## Reading order

1. **Introduction / methods:** cone → seed with shell → edible nut, with the study's sample size and collection period.
2. **Main result:** edible nut weight for equal starting cone weights. The reader can switch between 1 kg and 100 g. Average shares are 3.62% (light cones) and 4.05% (heavy cones), with standard errors 0.18 and 0.17 percentage points.
3. **A different comparison:** healthy nuts per individual cone. Means 76.1 and 112.9; standard errors 3.7 and 4.4. Rounded whole-number labels, unrounded bars.
4. **Damage:** 100-symbol pictures show the average empty/damaged share. Values 15.9% and 9.0%, rounded to 16 and 9 symbols; the rounding and uncertainty are disclosed.
5. **Detailed results:** navigate the original regression tree a question at a time. Every leaf and group size follows Figure 1. These are descriptions of observed groups, not a prediction service or evidence of a treatment effect.
6. **Discussion:** separate the observed association from the proposed watering/fertilizer intervention.

## Source and limitations

Verónica Loewe-Muñoz, Claudia Delard, Rodrigo del Río, Mónica Balzarini, and Dusan Gomory (2024). *Recommendations for increasing yield of the edible Pinus pinea L. pine nuts*. PLOS ONE. DOI: https://doi.org/10.1371/journal.pone.0300008. CC BY 4.0.

Data are manually transcribed from Table 4 and Figure 1, checked against the saved publisher XML and figure in `../article-tokens/`.

- Study overall: 560 cones, seven Chilean plantations, ten collection winters from 2010–2020 excluding 2016. Regression tree: 328 cones, as labelled in Figure 1. Do not imply these are the same sample size.
- Heavy group: cone weight >503 g; light group: <393 g. Middle group excluded from this weight comparison.
- Batch amounts rescale reported mean percentages; they are not measured 1 kg batches or guarantees for an individual batch.
- Fresh cone weight is compared with dried kernel weight. Anatomy illustrations are not to scale.
- Error whiskers are ±1 standard error, not confidence intervals or variation among individual cones.
- Follow Figure 1, not the apparent swapped 2.60%/4.04% ordering in its accompanying prose.
- No invented box plots: the raw distributions behind Figure 2 were not reconstructed from averages.
- Country comparisons and treatment claims are not visualized as controlled experiments: they are not such experiments.

## Check

```sh
uv run --with playwright playwright install chromium --only-shell
uv run pine-nut-visuals/check.py
```

The check launches its own isolated headless browser. It checks both units, all six tree paths, exact symbol counts, keyboard focus preservation, and horizontal overflow at four screen widths. Light, dark, and narrow-screen layouts were also inspected during development.
