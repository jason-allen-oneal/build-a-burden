import { find } from "../src/service";
export async function testFind(): Promise<void> {
  const result = await find([{ id: "one" }], "one");
  if (!result.ok || result.value.id !== "one") throw new Error("failed");
}
