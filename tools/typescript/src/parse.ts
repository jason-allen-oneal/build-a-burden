import ts from "typescript";

type Request = { filename?: string; source?: string };
type Diagnostic = { message: string; code: number; file?: string; line?: number; column?: number };

function flatten(message: ts.DiagnosticMessageChain | string): string {
  return typeof message === "string" ? message : ts.flattenDiagnosticMessageText(message, "\n");
}

function diagnostics(sourceFile: ts.SourceFile): Diagnostic[] {
  const result: Diagnostic[] = [];
  function visit(node: ts.Node): void { ts.forEachChild(node, visit); }
  visit(sourceFile);
  return result;
}

function main(): void {
  let request: Request;
  try { request = JSON.parse(require("fs").readFileSync(0, "utf8")) as Request; }
  catch (error) { process.stdout.write(JSON.stringify({ success: false, diagnostics: [{ message: String(error), code: -1 }] })); return; }
  const filename = (request.filename || "generated.ts").split(/[\\/]/).pop() || "generated.ts";
  const source = request.source || "";
  const file = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true, filename.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const options: ts.CompilerOptions = { noEmit: true, allowJs: false, target: ts.ScriptTarget.ES2022 };
  const host = ts.createCompilerHost(options);
  const original = host.getSourceFile;
  host.getSourceFile = (name, languageVersion, onError, shouldCreateNewSourceFile) => name === filename ? file : original.call(host, name, languageVersion, onError, shouldCreateNewSourceFile);
  const program = ts.createProgram([filename], options, host);
  const parseDiagnostics = program.getSyntacticDiagnostics().map((item: ts.Diagnostic): Diagnostic => {
    const start = item.start ?? 0;
    const position = file.getLineAndCharacterOfPosition(start);
    return { message: flatten(item.messageText), code: item.code, file: filename, line: position.line + 1, column: position.character + 1 };
  });
  process.stdout.write(JSON.stringify({ success: parseDiagnostics.length === 0, diagnostics: parseDiagnostics, ast: { kind: file.kind, statements: file.statements.length } }));
}

main();
