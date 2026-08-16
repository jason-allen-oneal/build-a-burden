import ts from "typescript";

type Request = { filename?: string; source?: string };
type Diagnostic = { message: string; code: number; file?: string; line?: number; column?: number };

function flatten(message: ts.DiagnosticMessageChain | string): string { return typeof message === "string" ? message : ts.flattenDiagnosticMessageText(message, "\n"); }
function format(item: ts.Diagnostic): Diagnostic {
  const file = item.file;
  const start = item.start ?? 0;
  const position = file && file.getLineAndCharacterOfPosition(start);
  return { message: flatten(item.messageText), code: item.code, ...(file ? { file: file.fileName } : {}), ...(position ? { line: position.line + 1, column: position.character + 1 } : {}) };
}

function main(): void {
  let request: Request;
  try { request = JSON.parse(require("fs").readFileSync(0, "utf8")) as Request; }
  catch (error) { process.stdout.write(JSON.stringify({ success: false, diagnostics: [{ message: String(error), code: -1 }] })); return; }
  const filename = (request.filename || "generated.ts").split(/[\\/]/).pop() || "generated.ts";
  const source = request.source || "";
  // The helper intentionally has no npm-installed React types or runtime. Preserve
  // TSX syntax and provide a minimal intrinsic-element surface so fixture
  // compilation tests syntax/type shape without executing a React toolchain.
  const effectiveSource = filename.endsWith(".tsx")
    ? `export {}; declare global { namespace JSX { interface IntrinsicElements { [elementName: string]: unknown; } } }\n${source}`
    : source;
  const options: ts.CompilerOptions = { noEmit: true, noResolve: true, strict: true, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.Preserve, skipLibCheck: true };
  const host = ts.createCompilerHost(options);
  const original = host.getSourceFile;
  host.getSourceFile = (name, languageVersion, onError, shouldCreateNewSourceFile) => name === filename ? ts.createSourceFile(name, effectiveSource, languageVersion, true) : original.call(host, name, languageVersion, onError, shouldCreateNewSourceFile);
  const program = ts.createProgram([filename], options, host);
  const all = [...program.getSyntacticDiagnostics(), ...program.getSemanticDiagnostics()].map(format);
  process.stdout.write(JSON.stringify({ success: all.length === 0, exit_status: all.length === 0 ? 0 : 1, diagnostics: all, timeout: false, resource_limit: false }));
}

main();
