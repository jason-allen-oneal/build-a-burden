export interface HasName { readonly name: string }
export class User implements HasName {
  constructor(public readonly name: string, private readonly roles: readonly string[] = []) {}
  can(role: string): boolean { return this.roles.includes(role); }
}
