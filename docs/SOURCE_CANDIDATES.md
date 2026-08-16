# Source Candidate Inventory

Discovery snapshot for the next corpus-intake decision. This file is **not** an
approval record and does not authorize cloning, downloading, or corpus
admission. A source enters the corpus only through
`manifests/approved-sources.jsonl` after its exact revision, license files,
scope, and provenance are reviewed.

## Discovery metadata

- Retrieved: `2026-08-14T18:05:05Z`
- Method: read-only GitHub REST repository/languages/commit metadata
- Search connector: Firecrawl was attempted first but returned HTTP 402 (no
  available search credits); no repository contents were cloned or executed.
- License values below are GitHub's detected SPDX metadata, not a legal
  conclusion. Verify the license files at the frozen revision before approval.
- Language percentages are GitHub's byte-based language statistics. The corpus
  pipeline still applies extension, parser, generated-code, secret, quality,
  and provenance filters.

## Tier A: initial review candidates

These have an allowlisted repository-level SPDX result and a TypeScript-first
language profile. They remain unapproved until Jason/designated reviewer signs
off on a frozen revision and scope.

As of `2026-08-15`, seven sources are approved and acquired at exact revisions:
`colinhacks-zod`, `vitejs-vite`, `nestjs-nest`, `typescript-eslint-typescript-eslint`,
`typeorm-typeorm`, `reduxjs-redux-toolkit`, and `microsoft-typescript`. The remaining
rows are still candidates only.

| Source ID | Repository | License metadata | TypeScript bytes | Default branch / observed HEAD | Recommended first scope |
|---|---|---:|---:|---|---|
| `microsoft-typescript` | https://github.com/microsoft/TypeScript | Apache-2.0 | 99.9% | `main` / `b465fdbfe175304d9b977da137b2c178ae1091d3` | Compiler, parser, checker, tests, and diagnostics; exclude build output, bundled files, and non-TS implementation languages. |
| `vitejs-vite` | https://github.com/vitejs/vite | MIT | 83.0% | `main` / `dcf88bd2ad2b1a8845f9029587cc8c825e382d42` | Core packages and TypeScript tests; exclude generated fixtures, lockfiles, and non-TS source. |
| `nestjs-nest` | https://github.com/nestjs/nest | MIT | 99.9% | `master` / `09dba60ce8dd47f9e6c518a86e2ac3cefdb6d68f` | Framework packages, examples, and tests after generated/vendor filtering. |
| `colinhacks-zod` | https://github.com/colinhacks/zod | MIT | 89.7% | `main` / `4e1720c80e65a6f2c8d1f9fc9da0ba3a1a4c9d86` | Runtime schemas, type-inference tests, and selected documentation context. |
| `typescript-eslint-typescript-eslint` | https://github.com/typescript-eslint/typescript-eslint | MIT | 91.6% | `main` / `b3f0cceee7b1d68f967253abc317f1b1b28fe4c8` | Parser, rules, utilities, and tests; exclude generated fixtures and bundled artifacts. |
| `typeorm-typeorm` | https://github.com/typeorm/typeorm | MIT | 99.8% | `master` / `df07bf1ef46f62f72077699fe4cd2c03b2f666e4` | ORM source and TypeScript tests; inspect database fixtures and credentials before admission. |
| `fuellabs-fuels-ts` | https://github.com/FuelLabs/fuels-ts | Apache-2.0 | 97.4% | `master` / `b3f37c91aca4aa9d5e4c0d3967f66237190826ea` | TypeScript SDK packages and tests; exclude Sway/Rust/non-TS implementation files. |
| `puppeteer-puppeteer` | https://github.com/puppeteer/puppeteer | Apache-2.0 | 93.9% | `main` / `8e1022b543f27d9d517310435aaa2b299fe99d46` | TypeScript API/client code and tests; exclude downloaded browser binaries, generated protocol data, and non-TS files. |

## Tier B: useful, but stage later

| Source ID | Repository | License metadata | TypeScript bytes | Default branch / observed HEAD | Why stage it |
|---|---|---:|---:|---|---|
| `microsoft-vscode` | https://github.com/microsoft/vscode | MIT | 95.9% | `main` / `79c23829131ed689743ce106e469a4e73f5918c3` | Excellent editor/repository-context data, but very large and includes third-party, generated, and native components. |
| `angular-angular` | https://github.com/angular/angular | MIT | 87.4% | `main` / `ec678c0f8d449b5be81438738d09f9f503f3e1a0` | Strong framework and compiler examples, but large multi-package tree requires a narrow path manifest first. |
| `storybookjs-storybook` | https://github.com/storybookjs/storybook | MIT | 85.3% | `next` / `7f28076f6b7f36e7675c6b52445b05dd98bda863` | Valuable TSX/component/test data, but substantial snapshots, docs, and generated content need review. |

## Review or exclude under the current policy

| Source ID | Repository | Observed license metadata | Decision |
|---|---|---|---|
| `microsoft-typescript-website` | https://github.com/microsoft/TypeScript-Website | CC-BY-4.0 | Documentation is useful, but CC-BY-4.0 is not in the current allowlist. Keep in review unless the data policy is explicitly amended. Observed `v2` HEAD: `0e7f8dcd2ca1f41d0d35c6e1a75d52290f8a705c`. |
| `definitelytyped-definitelytyped` | https://github.com/DefinitelyTyped/DefinitelyTyped | NOASSERTION | Do not admit as one repository. Package declarations carry heterogeneous licensing/provenance; use only with package-level review and records, if ever approved. Observed `master` HEAD: `72eec8e92e22f05da10dca6e23d9bc1ca2a56caf`. |

## Recommended next decision

For the next controlled expansion tranche, review these sources in order:

1. `FuelLabs/fuels-ts` — Apache-2.0, TypeScript-first SDK and tests.
2. `puppeteer/puppeteer` — Apache-2.0, substantial API/client and test patterns.
3. `angular/angular` — MIT, large framework/compiler source after narrow path review.
4. `storybookjs/storybook` — MIT, TSX/component patterns after snapshot filtering.
5. `microsoft/vscode` — MIT, valuable repository-context data but only after
   third-party and generated-content isolation.

`microsoft/TypeScript` is already approved in the current tranche after the
smaller-source intake controls were exercised. Do not treat observed HEADs as
permanent: freeze and re-check the commit, license files, and source scope
immediately before any future approval.

## Evaluated on 2026-08-15 — `alainbrown/openfable`

`https://github.com/alainbrown/openfable` is Apache-2.0 at observed `main`
commit `991a03545b5377dc57bc897d4e8cd96e0b9a971e`, but it is a Python retrieval
engine rather than a TypeScript corpus source (GitHub language metadata: roughly
249k Python bytes and 6k TypeScript bytes). It is therefore **not approved or
acquired** for the TypeScript base corpus. Its retrieval architecture may be
useful as a later research reference, but adding it would not materially improve
TypeScript completion data and would violate the project's narrow language
boundary if its Python implementation were admitted.
