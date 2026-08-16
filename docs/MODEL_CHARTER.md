# Model Charter

## Purpose

`ts-coder` is a TypeScript/TSX completion model trained from random initialization.
Its users are developers and researchers studying narrow, provenance-aware code
models. Intended tasks are causal completion, FIM completion, compiler-diagnostic
repair, tests-to-implementation, and repository-context completion.

The model is not a general assistant, autonomous agent, security authority, or
source of executable truth. It does not target JavaScript or other implementation
languages. Output should be TypeScript/TSX unless an evaluation explicitly asks for
an explanation. Users must review, compile, test, and security-check generated code.

Success means reproducible training, measurable syntax/compilation improvements,
useful repository context, bounded memorization, and no hidden pretrained foundation.

