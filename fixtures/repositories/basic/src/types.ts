export interface Identified { id: string }
export type Result<T> = { ok: true; value: T } | { ok: false; error: Error };
export type Named<T extends Identified> = T & { name: string };
