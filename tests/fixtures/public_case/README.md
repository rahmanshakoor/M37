# Public demo case

The hand-written case CONTRACTS.md calls the "public demo case": one sample,
`PUBLIC01`, twelve public alleles, GRCh38 with bare contig names (`1`, `7`, `15`,
`17`; `##contig` lines declared), every record `PASS`. It exercises stages 1–6 end to
end (`scripts/run_public_case.sh`) without any patient data and is the first
benchmark case for stage 7. **Nothing here derives from a person**: every variant is
a ClinVar/dbSNP entry, and the genotype fields (GT, AD, DP, GQ, PL, PGT, PID, QUAL,
INFO) are invented to look like a GATK single-sample call set.

## The alleles

Every row was verified live against Ensembl VEP REST (`POST /vep/human/region`,
Ensembl 116, GRCh38) on 2026-09-13; the reference base at each position was read from
`GET /sequence/region/human/<chrom>:<pos>..<pos>`; the two pathogenic CFTR alleles
were also confirmed in the release-pinned ClinVar VCF (`clinvar_20260905.vcf.gz`).
CFTR p.Gly542Ter's coordinates came from
`GET /vep/human/hgvs/NM_000492.4:c.1624G>T` (`7:117587778 G>T`, `stop_gained`).

| # | key (GRCh38)            | rsID        | gene  | consequence (MANE)                    | ClinVar (VCV)                       | AF (gnomAD v4, as VEP/the API report it) | GT  | role in the demo |
|---|-------------------------|-------------|-------|---------------------------------------|-------------------------------------|-------------------------------------------|-----|------------------|
| 1 | `1:11794419:T:G`        | rs1801131   | MTHFR | missense p.Glu429Ala (A1298C)         | Benign/Likely_benign (3521)         | 0.31         | 0/1 | common decoy; second het in MTHFR (two common hets must not form a comphet) |
| 2 | `1:11796321:G:A`        | rs1801133   | MTHFR | missense p.Ala222Val (C677T)          | drug_response (3520)                | 0.32         | 0/1 | the common decoy named in the brief |
| 3 | `7:117559403:A:G`       | rs34855237  | CFTR  | intron c.1393-61A>G                   | Benign (818106)                     | 0.23         | 0/1 | intronic decoy inside the F508del phasing group |
| 4 | `7:117559479:G:A`       | rs213950    | CFTR  | missense p.Val470Met (M470V)          | Benign/Likely_benign (7130)         | 0.43         | 0/1 | second common benign; same phasing group |
| 5 | `7:117559590:ATCT:A`    | rs113993960 | CFTR  | inframe deletion p.Phe508del          | Pathogenic, 4 stars (7105)          | 0.0119       | 0/1 | **pathogenic allele 1** (PGT `1|0`, PID `117559403_A_G`) |
| 6 | `7:117578521:G:A`       | rs213953    | CFTR  | intron                                | Benign, 1 star (1704415)            | 0.55         | 1/1 | deep intronic decoy, homozygous |
| 7 | `7:117587778:G:T`       | rs113993959 | CFTR  | stop_gained p.Gly542Ter               | Pathogenic, 4 stars (7115)          | 0.00036      | 0/1 | **pathogenic allele 2** (no PID) — no PID shared with allele 1 |
| 8 | `7:117589483:T:A`       | rs213965    | CFTR  | intron (deep; 1.7 kb from exon 12)    | Benign (53332)                      | 0.54         | 0/1 | the deep-intronic decoy named in the brief |
| 9 | `7:117595001:T:G`       | rs1042077   | CFTR  | synonymous p.Thr854=                  | Benign/Likely_benign (43577)        | 0.34         | 0/1 | the common synonymous decoy named in the brief |
|10 | `15:40185630:G:A`       | rs1801376   | BUB1B | missense p.Arg349Gln                  | Benign (133780)                     | 0.68         | 1/1 | common benign in a fourth gene, homozygous |
|11 | `17:7675088:C:T`        | rs28934578  | TP53  | missense p.Arg175His                  | Pathogenic, 3 stars (12374)         | 4.3e-6       | 0/1 | dominant pathogenic het (a `het_single` candidate that must rank below the CFTR pair) |
|12 | `17:7676154:G:C`        | rs1042522   | TP53  | missense p.Pro72Arg                   | Benign, 3 stars (12351)             | 0.72         | 0/1 | common decoy in the same gene as the TP53 het |

Rows 3–5 share the physical-phasing group `117559403_A_G` (76 and 111 bp apart —
within one read pair): rows 3 and 4 are `0|1`, p.Phe508del is `1|0`, so the deletion
is read as *trans* to the two common alleles. p.Gly542Ter (28.2 kb downstream) has no
PID, so the pathogenic pair's phase is `unknown` and stage 3 forms `CFTR:comphet`.

Expected stage-3 result (observed with `configs/filter.yaml` as shipped): 12 rows in,
3 kept, 9 dropped (`clinvar_benign` 4, `consequence:MODIFIER` 3,
`consequence:LOW` 1, `rarity` 1 — rs1801133, ClinVar `drug_response`, is not
Benign and falls to the AF rule); candidates `CFTR:comphet` (priority 1, both alleles,
p.Phe508del rescued from `recessive_max_af` by ClinVar P at 4 stars) then
`TP53:het_single` (priority 2).

## `case.yaml`

`proband_id: PUBLIC01`, the VCF, and five cystic-fibrosis HPO terms, each verified to
exist via `https://ontology.jax.org/api/hp/terms/<id>` on 2026-09-13: HP:0012236
Elevated sweat chloride, HP:0001738 Exocrine pancreatic insufficiency, HP:0002205
Recurrent respiratory infections, HP:0006528 Chronic lung disease, HP:0002110
Bronchiectasis. Exomiser 14.0.0 / 2406 data ranked CFTR first (AD 0.9963, AR
0.9894 with both alleles contributing) and TP53 third with these terms.

## `02_retrieve/` — recorded stage-2 output

`evidence/` is the stage-2 evidence store the live run wrote for the twelve variants
(Ensembl VEP 116 REST, `clinvar_20260905.vcf.gz`, gnomAD v4 API — 27 records:
12 vep, 12 clinvar, 3 gnomad, the gnomAD API being asked only for the three
consequential rare alleles), and `annotated_rows.json` is
`variants.annotated.tsv.gz` as a list of row dicts (73 columns, in the writer's order).
`tests/test_public_case.py` rebuilds the table with `gzip` (mtime 0) and runs the real
stage 3 and the stage-5 dry run on it without the network. Regenerate both with
`scripts/run_public_case.sh` and copy `02_retrieve/evidence/` plus the rows of
`engine.retrieve.run.read_annotated` from the run directory.

## `05_reason/` — recorded stage-5 chain

`chain_CFTR_comphet.json` is the chain the real stage 5 (`engine.reason.run.run_reason`,
`top_n=1`) wrote for `CFTR:comphet` over the run directory above — the recorded
stage-2 store, stage 3 as shipped, the HPO terms from `case.yaml`, no stage 4 — with
a scripted `FakeClient` and no literature turn, the way `tests/fixtures/medicine/`
was produced. So it has exactly the shape and key order stage 5 writes, every id it
cites is one of the pair's six stage-2 records (`vep:`, `clinvar:`, `gnomad:` for each
allele; `literature: []`, no PS3), the validator rejected and disputed nothing, and
the classifications are the engine's own from the criteria: p.Gly542Ter `pathogenic`
(PVS1, PM3, PP5, PP4; PM2 not met at AF 0.00036), p.Phe508del `likely_pathogenic`
(PM4, PM3, PP5, PP4; PM2 not met at AF 0.0119). It is a scripted client's chain, not
a model's opinion about this case.

Why it exists: a stage-5 dry run writes bundles and prompts, not chains, and stage 6
reports on a chain. `scripts/run_public_case.sh` therefore copies this file into
`<run>/05_reason/chains/CFTR:comphet.json` after the dry run — saying so on the
terminal, and leaving the stage-5 manifest honest (`dry_run: true`,
`chains_written: 0`) — so that `engine medicine --dry-run` runs on it and stages 1–6
are exercised end to end without a key. `PUBLIC_LIVE_MODEL=1` (with
`ANTHROPIC_API_KEY`) runs stage 5 against the model instead and stage 6's dry run on
the chain the model produced. `tests/test_public_case.py` checks the chain against the
recorded store and runs the stage-6 dry run on it without the network.

## Files

- `public.vcf` — the readable source; `public.vcf.gz` + `.tbi` — `bgzip -c public.vcf`
  and `tabix -p vcf`, what `case.yaml` points at.
- `05_reason/chain_CFTR_comphet.json` — the recorded stage-5 chain above.
- `.gitignore` (this directory) — the repository ignores `*.vcf`, `*.vcf.gz`,
  `*.vcf.gz.tbi` and `case.yaml` everywhere by design (patient data must never be
  committed); this nested file re-includes exactly the four public files above and
  nothing else — the patterns are anchored (`!/public.vcf` …), so the same names in a
  subdirectory of this one stay ignored, and `git check-ignore` still refuses a VCF
  or case file anywhere else.
