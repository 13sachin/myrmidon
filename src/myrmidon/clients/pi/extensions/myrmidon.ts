/**
 * Myrmidon's presence inside Pi: harness code beside the person.
 *
 * Through MCP the harness only ever hears tool calls, which are the model's. What is the
 * person's, or the harness's own to check, happens here, in Pi's process:
 *
 * - the approval: when `mandate_write` returns a mandate that is ready, the person is asked
 *   in the chat, and their yes runs `myrmidon approve` as them; a shell spelling of the
 *   same act is blocked. Without a UI (`pi -p`) it fails closed and names the terminal
 *   verb;
 * - the look at the start of a session: where the study stands, as context for the first
 *   run, and the study's one-line state in the footer;
 * - the views: /status, /plan, /inbox, /approve, /answer;
 * - the provider: every endpoint in endpoints.toml, registered at load, the key read from
 *   the secrets file by a command, never written anywhere.
 *
 * The extension holds no logic about mandates or findings: the CLI renders, the CLI
 * records, the server checks, and all run from the install named in `.pi/mcp.json`.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

const SERVER = "myrmidon";
const TOOL = (name: string) => `mcp__${SERVER}__${name}`;
const MANDATE_TOOL = TOOL("mandate_write");
/** A shell spelling of the approval, the person's act: `myrmidon approve` however the
 *  words are tokenised (a shell line, a list of words, `python -m myrmidon.cli`). Reading
 *  the ledger or the source is not the act and passes. */
const SHELL_APPROVAL = /\bmyrmidon(?:\.cli)?\b[^\w\n]*\bapprove\b/;
/** Pi's own process carries PI_CODING_AGENT too, so every call names its channel. */
const VIA = ["--via", "pi-extension"];
/** How often the footer may ask the harness for the study's state. */
const STATUS_EVERY_MS = 10_000;

interface ServerEntry {
  command: string;
  args: string[];
}

interface ApprovalView {
  study: string;
  digest: string;
  question: string;
  estimate_core_h?: number;
  ceiling_core_h: number;
  spent_core_h?: number;
  revision?: number;
  blocked?: string;
  approved?: boolean;
  text: string;
}

interface EndpointsView {
  secrets: string;
  endpoints: {
    name: string; api_base: string; key_env: string; key_set: boolean;
    context_window?: number | null; models: string[];
  }[];
}

// ---------------------------------------------------------------- the install

function readServers(path: string): Record<string, ServerEntry> {
  if (!existsSync(path)) return {};
  try {
    const data = JSON.parse(readFileSync(path, "utf8"));
    return (data && typeof data === "object" && data.mcpServers) || {};
  } catch {
    return {};
  }
}

/** The server entry Pi starts for this project: project `.pi/mcp.json`, then the user's. */
function serverEntry(cwd: string): ServerEntry | undefined {
  const agentDir = process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
  for (const path of [join(cwd, ".pi", "mcp.json"), join(agentDir, "mcp.json")]) {
    const entry = readServers(path)[SERVER];
    if (entry && typeof entry.command === "string") {
      return { command: entry.command, args: Array.isArray(entry.args) ? entry.args.map(String) : [] };
    }
  }
  return undefined;
}

/** The myrmidon command line in the server's world: the program the server entry names,
 *  then the command. The entry names no study — the server finds the project from its
 *  working directory, and so does each command, run where the session is. */
function cli(entry: ServerEntry, command: string[]): string[] {
  const at = entry.args.indexOf("serve");
  const program = at < 0 ? entry.args : entry.args.slice(0, at);
  return [entry.command, ...program, ...command];
}

function run(argv: string[], cwd: string, timeout = 120_000): { status: number; stdout: string; stderr: string } {
  const out = spawnSync(argv[0], argv.slice(1), { cwd, encoding: "utf8", timeout });
  return { status: out.status ?? 1, stdout: out.stdout ?? "", stderr: out.stderr ?? "" };
}

// ---------------------------------------------------------------- the approval

/** What the dialog itself shows: a few lines a terminal of any height holds, with the
 *  full mandate already in the transcript above it. */
function dialogText(view: ApprovalView): string {
  const estimate = view.estimate_core_h ? `estimated ${view.estimate_core_h} core-h; ` : "";
  const spent = view.spent_core_h ? `, ${view.spent_core_h} already booked` : "";
  return [view.question, "",
    `cost      ${estimate}at most ${view.ceiling_core_h} core-h${spent}`,
    `revision  ${view.revision}`,
    `digest    ${view.digest.slice(0, 12)} — the approval is recorded against this`].join("\n");
}

const TERMINAL_VERB =
  "Approving is the researcher's act. Say the mandate's cost and digest and end the turn; " +
  "they approve at a terminal with `myrmidon approve`.";

/** The approval, as a person gives it in the chat. Returns what the model is told. */
async function approveInChat(pi: ExtensionAPI, ctx: ExtensionContext): Promise<{ reason: string; approved: boolean }> {
  const entry = serverEntry(ctx.cwd);
  if (!entry) {
    return { approved: false, reason: `no ${SERVER} server entry found for this project; ${TERMINAL_VERB}` };
  }
  if (!ctx.hasUI) {
    // print and JSON modes: nobody can be asked, so the gate fails closed
    return { approved: false, reason: `nobody is at the keyboard in this mode. ${TERMINAL_VERB}` };
  }
  const approve = cli(entry, ["approve"]);
  const shown = run([...approve, "--show", "--json", ...VIA], ctx.cwd);
  if (shown.status !== 0) {
    // the view names what blocks it; that, not the view, is the line the researcher reads
    let blocks: string[] = [];
    try {
      const refused: ApprovalView = JSON.parse(shown.stdout);
      blocks = [refused.blocked ?? ""].filter(Boolean);
    } catch {
      // not a view: the command's own words below say why
    }
    const why = blocks.length ? blocks.join("\n") : (shown.stdout + shown.stderr).trim();
    return { approved: false, reason: `it cannot be approved as it stands: ${why}` };
  }
  let view: ApprovalView;
  try {
    view = JSON.parse(shown.stdout);
  } catch {
    return { approved: false, reason: `myrmidon approve --show returned no readable view:\n${shown.stdout.slice(0, 2000)}` };
  }
  // the mandate in full goes into the transcript, where it scrolls; the dialog stays short.
  // The model reads this message too, so it says what it is: a rendering, not an approval
  if (view.approved) {
    // a second result for a mandate already approved: the researcher is not asked twice
    return { approved: true, reason: `mandate revision ${view.revision} is approved already.` };
  }
  pi.sendMessage({
    customType: `${SERVER}.mandate`,
    content:
      "The mandate as the approval dialog shows it to the researcher. This message approves " +
      "nothing; the researcher's answer follows it.\n\n" + view.text,
    display: true,
  });
  const title = `Approve mandate revision ${view.revision}, at most ${view.ceiling_core_h} core-h?`;
  const yes = await ctx.ui.confirm(title, dialogText(view));
  if (!yes) {
    return {
      approved: false,
      reason:
        `the researcher did not approve the mandate (digest ${view.digest.slice(0, 12)}). ` +
        "Ask what they would change, or end the turn.",
    };
  }
  const done = run([...approve, "--study", view.study, "--digest", view.digest, "--yes", ...VIA, "--json"], ctx.cwd);
  if (done.status !== 0) {
    const why = (done.stdout + done.stderr).trim();
    return { approved: false, reason: `the approval was not recorded:\n${why}` };
  }
  let result: { event: string; approver: string; summary: string };
  try {
    result = JSON.parse(done.stdout.trim().split("\n").pop() ?? "");
  } catch {
    return { approved: false, reason: `approval recorded, but its result was unreadable:\n${done.stdout.slice(0, 2000)}` };
  }
  ctx.ui.notify(`approved by ${result.approver} — ${result.summary}`, "info");
  return {
    approved: true,
    reason: `Approved by ${result.approver} (event ${result.event}): ${result.summary}. ` +
      "Carry the stages out call by call under it; each submission is checked against it.",
  };
}

/** Whether a `mandate_write` result says the mandate is ready for the researcher: the
 *  observation's JSON in the text Pi shows, or in the raw result when that text was cut. */
function mandateReady(content: { type: string; text?: string }[], raw?: unknown): boolean {
  const texts = content.filter((p) => p.type === "text" && p.text).map((p) => p.text as string);
  const shaped = raw as { content?: { text?: string }[]; data?: { status?: string } } | undefined;
  if (shaped?.data?.status === "ready") return true;
  const full = shaped?.content ?? [];
  for (const text of [...texts, ...full.map((p) => p.text ?? "")]) {
    try {
      const result = JSON.parse(text);
      // a tool's own fields arrive under `data`
      if (result && result.data?.status === "ready" && result.ok !== false) return true;
    } catch {
      // not the observation's JSON as a whole: a cut text still names its status
      if (/"ok":\s*true/.test(text) && /"status":\s*"ready"/.test(text)) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- the extension

export default function (pi: ExtensionAPI) {
  let looked = false;
  let statusAt = 0;

  // -- the provider: every endpoint, the key read from the secrets file by a command
  const here = serverEntry(process.cwd());
  if (here) {
    const listed = run([...cli(here, ["endpoints"]), "--json"], process.cwd(), 30_000);
    if (listed.status === 0) {
      try {
        const view: EndpointsView = JSON.parse(listed.stdout);
        for (const ep of view.endpoints) {
          if (!ep.models.length) continue;
          pi.registerProvider(ep.name, {
            name: `${ep.name} (myrmidon endpoints.toml)`,
            baseUrl: ep.api_base,
            apiKey: `!sh -c '. ${JSON.stringify(view.secrets)} 2>/dev/null; printf %s "$${ep.key_env}"'`,
            api: "openai-completions",
            models: ep.models.map((id) => ({
              id,
              name: id,
              reasoning: false,
              input: ["text"],
              contextWindow: ep.context_window ?? 128_000,
              maxTokens: 16_384,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            })),
          });
        }
      } catch {
        // an unreadable listing registers nothing; Pi's own providers stand
      }
    }
  }

  const refreshStatus = (ctx: ExtensionContext, force = false) => {
    if (!ctx.hasUI) return;
    const now = Date.now();
    if (!force && now - statusAt < STATUS_EVERY_MS) return;
    statusAt = now;
    const entry = serverEntry(ctx.cwd);
    if (!entry) return;
    const out = run(cli(entry, ["statusline"]), ctx.cwd, 20_000);
    if (out.status === 0 && out.stdout.trim()) ctx.ui.setStatus(SERVER, out.stdout.trim());
  };

  const show = (ctx: ExtensionContext, command: string[]) => {
    const entry = serverEntry(ctx.cwd);
    if (!entry) {
      ctx.ui.notify(`no ${SERVER} server entry found for this project`, "error");
      return;
    }
    const out = run(cli(entry, command), ctx.cwd);
    const text = (out.stdout + (out.status === 0 ? "" : "\n" + out.stderr)).trim() || "(nothing)";
    pi.sendMessage({ customType: `${SERVER}.${command[0]}`, content: text, display: true });
  };

  // -- the session: the footer at start, the look at the first run
  pi.on("session_start", async (_event, ctx) => {
    looked = false;
    refreshStatus(ctx, true);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (event.messageType !== "user") return undefined;
    if (looked) return undefined;
    looked = true;
    const entry = serverEntry(ctx.cwd);
    if (!entry) return undefined;
    // `study open`: what ended while nobody watched, what waits, where the study stands
    const out = run(cli(entry, ["study", "open"]), ctx.cwd, 60_000);
    const text = out.stdout.trim();
    if (!text) return undefined;
    return {
      message: {
        customType: `${SERVER}.study`,
        content: `Where the study stands, as the harness sees it before this session:\n${text}`,
        display: true,
      },
    };
  });

  // -- the gate
  pi.on("tool_call", async (event) => {
    if (event.toolName !== "bash") return undefined;
    const command = String((event.input as { command?: unknown }).command ?? "");
    if (SHELL_APPROVAL.test(command)) {
      return { block: true, reason: `a shell command is not the researcher's yes. ${TERMINAL_VERB}` };
    }
    return undefined;
  });

  // -- the approval a ready mandate opens, and the footer after a harness call
  pi.on("tool_result", async (event, ctx) => {
    if (!event.toolName.startsWith(`mcp__${SERVER}__`)) return undefined;
    if (event.toolName === MANDATE_TOOL && !event.isError && ctx.hasUI
        && mandateReady(event.content, event.structuredContent)) {
      const outcome = await approveInChat(pi, ctx);
      refreshStatus(ctx, true);
      const text = `The researcher answered in the approval dialog: ${outcome.reason}`;
      return { content: [...event.content, { type: "text", text }] };
    }
    refreshStatus(ctx);
    return undefined;
  });

  // -- the views
  pi.registerCommand("status", {
    description: "where the study stands: the mandate, the jobs, what waits on you",
    handler: async (_args, ctx) => show(ctx, ["status"]),
  });
  pi.registerCommand("plan", {
    description: "this study's mandate and its trace: what it set out to do and what it did",
    handler: async (_args, ctx) => show(ctx, ["plan", "show"]),
  });
  pi.registerCommand("inbox", {
    description: "the questions that wait on you, with their options",
    handler: async (_args, ctx) => show(ctx, ["inbox"]),
  });
  pi.registerCommand("answer", {
    description: "answer a parked question: /answer ID OPTION",
    handler: async (args, ctx) => {
      const parts = args.trim().split(/\s+/).filter(Boolean);
      if (parts.length !== 2) {
        ctx.ui.notify("usage: /answer ID OPTION (the ids are in /inbox)", "warning");
        return;
      }
      show(ctx, ["answer", ...parts]);
    },
  });
  pi.registerCommand("approve", {
    description: "read this study's mandate and approve it — the dialog a ready mandate opens",
    handler: async (_args, ctx) => {
      const outcome = await approveInChat(pi, ctx);
      if (outcome.approved) {
        // the researcher's yes reaches the agent waiting on it, as their typed answer would
        pi.sendMessage({ customType: `${SERVER}.approval`, content: outcome.reason, display: true },
          { triggerTurn: true });
      } else {
        ctx.ui.notify(outcome.reason.split("\n")[0], "warning");
      }
      refreshStatus(ctx, true);
    },
  });
}
