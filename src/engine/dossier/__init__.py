"""Stage 5, second step — a gene dossier the chain and the medicine report can cite.

``engine dossier`` puts the reviewed UniProt entry of the candidate's gene
(:mod:`engine.retrieve.uniprot`) and a fixed set of Europe PMC searches into the
stage-5 store *before* any model runs, then asks the dossier agent
(:mod:`engine.dossier.run`, under the frozen instructions of
:mod:`engine.dossier.prompts`) what is known about the protein, the disease
mechanism, the regions the chain's variants fall in, published genotype patterns
and what a functional test would show. The engine — never the model — fills each
variant's residue and the features covering it (:mod:`engine.dossier.checks`), and
:mod:`engine.agents.validator` decides which claims survive. See CONTRACTS.md.
"""
