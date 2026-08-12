# Can gene set scores measure individual functional traits?

Analysis code for the preprint *Can gene set scores measure individual functional traits? Separating condition effects from individual-level coherence, and attributing the coherence that remains*.

Gene set scores were designed for group comparison. They are now computed per patient and used to order individuals by score. That shift requires something group comparison never has to check: that the genes in a set rise and fall together from one individual to the next. This repository contains the code that tests whether they do, in five public transcriptome cohorts.

**Preprint**: [DOI to be added]

---

## What the analysis measures

For every gene set, two properties are measured separately in the same individuals:

- **Condition effect** — does the score change when the same individual is perturbed?
- **Individual-level coherence** — do the genes in the set covary across individuals?

Every metric is evaluated against **random gene sets matched for size and mean expression**. Absolute values are not interpretable on their own: section 3.8 of the paper shows that the level of the controls moves 70-fold with the choice of normalisation alone.

Main result: condition effects are near-universal (88.3–90.8% of sets across two perturbation cohorts) while coherence is distinctly rarer, and whatever coherence appears tracks the largest source of variance in that experiment — measurement chip, blood cell composition, or ischemic time — rather than the biology the set is named for.

---

## Data

No data files are included. All inputs are public accessions, downloaded by `pixi run download`:

| Accession | Content | Source |
|---|---|---|
| E-MTAB-2232 | CD14+ monocytes, resting / LPS / IFN-γ, Illumina HT-12 v4 | ArrayExpress |
| GSE47353 | PBMC, influenza vaccination timecourse, Affymetrix Gene 1.0 ST | GEO |
| GSE35846 | Whole blood with technical and biological covariates | GEO |
| GSE81046 | Primary macrophages, RNA-seq, bacterial infection | GEO |
| GTEx v8 | Whole blood, RNA-seq | GTEx Portal |
| CORUM, Reactome, MSigDB Hallmark, TRRUST, PanglaoDB | Gene set libraries | via Enrichr |

Retrieval source, date and SHA-256 for every file are recorded in `data/checksums/provenance.jsonl`, which **is** included. That record is the only way a third party can confirm they obtained the same bytes.

GTEx data are used under the terms of the GTEx Portal. The Genotype-Tissue Expression (GTEx) Project was supported by the Common Fund of the Office of the Director of the National Institutes of Health, and by NCI, NHGRI, NHLBI, NIDA, NIMH and NINDS.

---

## Reproducing the analysis

The environment is defined by [pixi](https://pixi.sh); `pixi.lock` pins every dependency.

```bash
pixi install
pixi run test          # validate the metrics on synthetic data (11 tests)
pixi run download      # fetch public data and gene sets, record SHA-256
pixi run preprocess    # build gene x individual matrices per condition
pixi run modules       # derive co-expression modules from discovery donors only
pixi run validation    # build the timepoint matrices for GSE47353
pixi run analyze       # gene set x metric table
pixi run attribution   # decompose where the coherence comes from
pixi run batch         # test whether PCs are explained by chip, position, batch
pixi run retest        # test-retest reliability (ICC) and external reproducibility
pixi run phenotype     # phenotype association and its reproducibility across draws
pixi run technical     # technical vs biological vs composition in whole blood
pixi run sensitivity   # coverage filter sensitivity
pixi run figures       # generate the figures
```

`pixi run all` runs the chain end to end.

RNA-seq cohorts reuse **the same analysis code**. Only the input/output namespace and the configuration file are switched, via the `T26_DATASET` environment variable:

```bash
pixi run wp1-preprocess          # GSE81046
pixi run wp1-modules
pixi run wp1-analyze
pixi run wp1-attribution
pixi run wp1-ancestry
pixi run wp1-normalization       # Table 4: effect of quantification and normalisation
pixi run wp1-filter-sensitivity  # Table 7: sensitivity to the expression filter
pixi run wp1-gtex-preprocess     # GTEx v8 whole blood
pixi run wp1-gtex-attribution
```

This matters for the claim: if the code had to be adjusted per cohort, "the result reproduced in another dataset" would mean much less. Differences in results derive from differences in the data alone.

---

## Layout

```
src/            analysis code (shared across all cohorts)
  preprocessing/  matrix construction, TMM, co-expression modules
  scoring/        z-mean, singscore, PLAGE, median rank
  reliability/    all metrics, controls, attribution
  visualization/  figures, manuscript conversion
tools/          numerical audit of the manuscript, detector report reader
tests/          validation of the metrics on synthetic data
config/         all thresholds, seeds and cohort definitions
results/tables/   every number in the paper, one row per gene set
results/figures/  the figures, PNG and SVG
data/checksums/   retrieval provenance for every input file
```

Random seeds, gene set retrieval dates and sample exclusion criteria are fixed in `config/`, not in the code.

---

## Verifying the numbers in the paper

```bash
pixi run audit
```

This recomputes every quantitative claim from `results/tables/` and checks that the value appears in the manuscript. It works in that direction on purpose: transcribing the manuscript's numbers into the checking script would make the transcription itself a new source of error. It also holds a list of retired values that must no longer appear anywhere, which catches the case where one section was corrected and another was not.

---

## License

Code is released under the MIT License (see `LICENSE`). The public datasets remain under the terms of their respective sources.

---

## Citation

If you use this code, please cite the preprint. [Citation to be added]
