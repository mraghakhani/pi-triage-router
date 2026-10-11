/** Captures the virtual model registered by the extension's default export. */
import * as router from "../triage-router.ts";
import type { FakeModel } from "./fixtures.ts";

export interface CapturedVirtualModel {
	provider: string;
	id: string;
	name: string;
	route: (request: any, ctx: any) => Promise<{ model: FakeModel; thinkingLevel: unknown; state?: unknown }>;
}

export function captureRoute(): CapturedVirtualModel {
	return captureModels()[captureModels().length - 1];
}

/** All virtual models registered by the extension (laya/auto, maybe jev/auto). */
export function captureModels(): CapturedVirtualModel[] {
	const all: CapturedVirtualModel[] = [];
	const pi = {
		registerVirtualModel(def: Record<string, unknown>) {
			all.push({ ...def } as unknown as CapturedVirtualModel);
		},
	};
	router.default(pi as never);
	if (!all.length) throw new Error("extension did not register a virtual model");
	return all;
}