"""Stage 6 — from a validated evidence chain to drug hypotheses, cited by record id.

Four gene-centric retrievers (:mod:`~engine.medicine.opentargets`,
:mod:`~engine.medicine.dgidb`, :mod:`~engine.medicine.chembl`,
:mod:`~engine.medicine.ctgov`) turn public drug, interaction, mechanism and trial data
into :class:`~engine.retrieve.store.EvidenceRecord` objects. The medicine agent
(:mod:`~engine.medicine.run`) reads the stage-5 chain, argues from the records the
five tools (:mod:`~engine.medicine.tools`) return under a frozen instruction set
(:mod:`~engine.medicine.prompts`), and :mod:`engine.agents.validator` decides what
survives — every drug candidate must rest on a record about the drug and carry its
counter-arguments. See CONTRACTS.md.
"""
