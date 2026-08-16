import ts from "typescript";

export type Span = { kind: string; start: number; end: number };
export function statementSpans(source: string, filename = "generated.ts"): Span[] {
  const file = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  return file.statements.map((statement) => ({ kind: ts.SyntaxKind[statement.kind], start: statement.getStart(file), end: statement.end }));
}

if (require.main === module) {
  try {
    const request = JSON.parse(require("fs").readFileSync(0, "utf8")) as { source?: string; filename?: string };
    const source = request.source ?? "";
    process.stdout.write(JSON.stringify({ success: true, statements: statementSpans(source, request.filename) }));
  } catch (error) {
    process.stdout.write(JSON.stringify({ success: false, diagnostics: [{ message: String(error), code: -1 }] }));
  }
}
