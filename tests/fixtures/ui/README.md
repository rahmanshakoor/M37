# UI test fixtures

What `tests/test_ui.py` builds a temporary **project directory** from — the layout
`engine ui` discovers without options (CONTRACTS.md, "UI — one-click pipeline"):

    <project>/
      case.yaml            ← case.public.yaml with `vcf:` rewritten to the absolute public VCF
      case.broken.yaml     ← listed by /api/cases with its error, never hidden
      ref/                 ← stub `clinvar_<date>.vcf.gz` + `.tbi`, `funnel*.bed`, `exomiser/` as a test asks
      cache/
      work/                ← the runs the page makes

The reference stubs are empty files with the right names (discovery reads names,
indexes and the Exomiser `.verified.json` sidecars, never the bulk data); the stub
runner never executes them. The one real smoke test skips every stage that would
need the network and runs stages 3, 5 and 6 as real subprocesses with the scripted
`fake` provider. Nothing here derives from a person.
