import type { Identified, Result } from "./types";
export async function find<T extends Identified>(values: T[], id: string): Promise<Result<T>> {
  try {
    const value = values.find((item) => item.id === id);
    return value ? { ok: true, value } : { ok: false, error: new Error("missing") };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error : new Error(String(error)) };
  }
}
