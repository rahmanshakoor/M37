"""Stage 5 — reason over retrieved evidence, cite by record id.

The reasoning agent (:mod:`engine.reason.run`) argues from the bundle and three tools
(:mod:`engine.reason.tools`) under a frozen instruction set (:mod:`engine.reason.prompts`);
:mod:`engine.agents.validator` decides what survives, and the engine — never the
model — turns the surviving criteria into a classification. See CONTRACTS.md.
"""
