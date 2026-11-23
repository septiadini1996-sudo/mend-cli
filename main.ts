// mend: one patch tool for every AI edit format (unified diff, SEARCH/REPLACE blocks, V4A patches and JSON
// tool calls), written to SPEC.md in the starter kit. Node 22, standard library only.
//
// The program keeps the whole tree in memory while the edits of one patch run: each edit reads and writes a
// virtual tree, a failed edit is rolled back, and only when every edit succeeded is anything written to disk
// (all-or-nothing, SPEC.md section 10). Output lines and exit codes follow section 2 exactly.
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and then this file hands over to
// runner.ts (the last lines below): that hand-over and runner.ts stay as the kit ships them.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMAT_NAMES = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** An exit-2 outcome: the single line to print. */
class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}

/** One edit failed: its code and the path the failure concerns. */
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const isBlank = (s: string): boolean => s.trim() === '';

/** A patch's lines (section 3): split at \n, one trailing \r removed from each, no empty piece after a final \n. */
function patchLines(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n');
  if (text.endsWith('\n')) parts.pop();
  return parts.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

/** A path is safe (section 3) when it is not empty, not absolute, holds no backslash and no empty, `.` or `..` segment. */
function safePath(p: string): boolean {
  if (p === '' || p.startsWith('/') || p.includes('\\')) return false;
  return p.split('/').every((seg) => seg !== '' && seg !== '.' && seg !== '..');
}

interface FileModel {
  lines: string[];
  br: '\n' | '\r\n';
  final: boolean; // the text ends with a line break
}

/** A file's lines (section 3): its break is CRLF when its first line feed follows a carriage return, else LF. */
function splitText(text: string): FileModel {
  if (text === '') return { lines: [], br: '\n', final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const final = text.endsWith(br);
  const lines = text.split(br);
  if (final) lines.pop();
  return { lines, br, final };
}

/** The bytes of a file model; a file of no lines is written empty. */
function joinText(m: FileModel): string {
  if (m.lines.length === 0) return '';
  return m.lines.join(m.br) + (m.final ? m.br : '');
}

/**
 * The tree as the edits leave it. `now` holds the paths the patch has touched (null: deleted); anything else is
 * read from disk once and cached in `orig`. Successful move edits are remembered in `moves` for the `R` lines of
 * section 2; a failed edit rolls both maps back (the runner in `run` truncates `moves` to its saved length).
 */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  readonly moves: Array<readonly [string, string]> = [];

  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }

  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }

  /** The current state of a path once the whole patch has run. */
  final(p: string): string | null {
    return this.get(p);
  }
}

type Edit = (tree: Tree) => void;

// ---------------------------------------------------------------------------
// Section 5: matching lines at three tiers.
// ---------------------------------------------------------------------------

const stripTrail = (s: string): string => s.replace(/[ \t]+$/, '');
const stripBoth = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
const leadWs = (s: string): string => /^[ \t]*/.exec(s)![0]!;

function lineEqAt(a: string, b: string, tier: number): boolean {
  if (tier === 0) return a === b;
  if (tier === 1) return stripTrail(a) === stripTrail(b);
  return stripBoth(a) === stripBoth(b);
}

/** Every position i >= from where old's lines match the file's lines i, i+1, ... at the given tier. */
function occurrencesAt(lines: string[], old: string[], tier: number, from: number): number[] {
  const out: number[] = [];
  for (let i = Math.max(0, from); i + old.length <= lines.length; i += 1) {
    let ok = true;
    for (let j = 0; j < old.length; j += 1) {
      if (!lineEqAt(lines[i + j]!, old[j]!, tier)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

interface Match {
  tier: number;
  positions: number[];
}

/** The first tier at which old occurs at or after `from`; with `endsAtEnd`, only occurrences ending at the last line count. */
function matchOld(lines: string[], old: string[], strict: boolean, from: number, endsAtEnd = false): Match | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const tier of tiers) {
    let positions = occurrencesAt(lines, old, tier, from);
    if (endsAtEnd) positions = positions.filter((p) => p + old.length === lines.length);
    if (positions.length > 0) return { tier, positions };
  }
  return null;
}

/** The first line at or after `from` equal to a single-line anchor, at the first tier where one is; -1 when none. */
function matchAnchor(lines: string[], anchor: string, strict: boolean, from: number): number {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const tier of tiers) {
    for (let i = Math.max(0, from); i < lines.length; i += 1) {
      if (lineEqAt(lines[i]!, anchor, tier)) return i;
    }
  }
  return -1;
}

/**
 * One line of N in a hunk or chunk: a context line (written as the file's own matched line, by its index in O)
 * or an added line (written as it is, re-indented at tier 2).
 */
type NeuLine = { ctx: number } | { add: string };

/** Build the lines that replace the matched ones: context lines come from the file, added lines as written (tier 2 re-indents). */
function renderNeu(lines: string[], pos: number, old: string[], neu: NeuLine[], tier: number): string[] {
  const f = tier === 2 ? leadWs(lines[pos] ?? '') : '';
  const p = tier === 2 ? leadWs(old[0] ?? '') : '';
  return neu.map((it) => {
    if ('ctx' in it) return lines[pos + it.ctx]!;
    const s = it.add;
    if (tier !== 2) return s;
    if (s.trim() === '') return s;
    return s.startsWith(p) ? f + s.slice(p.length) : s;
  });
}

/** The body of a unified hunk or a V4A chunk: context, old and added lines, and `\` markers (unified only). */
type BodyLine = { k: 'ctx' | 'del' | 'add'; text: string } | { k: 'bs' };

function bodyOld(body: BodyLine[]): string[] {
  const out: string[] = [];
  for (const b of body) if (b.k === 'ctx' || b.k === 'del') out.push(b.text);
  return out;
}

function bodyNeu(body: BodyLine[]): NeuLine[] {
  const out: NeuLine[] = [];
  let oi = 0;
  for (const b of body) {
    if (b.k === 'ctx') {
      out.push({ ctx: oi });
      oi += 1;
    } else if (b.k === 'del') {
      oi += 1;
    } else if (b.k === 'add') {
      out.push({ add: b.text });
    }
  }
  return out;
}

/** True when a `\` line immediately follows the body's last context or `+` line (section 6's end-of-file rule). */
function noEolAfterLastCtxOrAdd(body: BodyLine[]): boolean {
  for (let i = body.length - 1; i >= 0; i -= 1) {
    const b = body[i]!;
    if (b.k === 'ctx' || b.k === 'add') return i + 1 < body.length && body[i + 1]!.k === 'bs';
  }
  return false;
}

/** True when a `\` line immediately follows the body's last `+` line (a created file's ending, section 6). */
function noEolAfterLastAdd(body: BodyLine[]): boolean {
  for (let i = body.length - 1; i >= 0; i -= 1) {
    if (body[i]!.k === 'add') return i + 1 < body.length && body[i + 1]!.k === 'bs';
  }
  return false;
}

// ---------------------------------------------------------------------------
// Section 6: unified diffs.
// ---------------------------------------------------------------------------

interface Hunk {
  a: number; // the A of @@ -A[,B] +C[,D] @@
  body: BodyLine[];
}

interface Section {
  diffOld?: string;
  diffNew?: string;
  minus?: string;
  plus?: string;
  renameFrom?: string;
  renameTo?: string;
  newFile: boolean;
  deletedFile: boolean;
  hunks: Hunk[];
}

const DIFF_RE = /^diff --git a\/(.+?) b\/(.+)$/;
const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

/** The path of a `--- `/`+++ ` line: after the four characters, up to the first tab, spaces removed, a/ or b/ stripped. */
function headerPath(line: string): string {
  let s = line.slice(4);
  const tab = s.indexOf('\t');
  if (tab >= 0) s = s.slice(0, tab);
  s = s.trim();
  if (s.startsWith('a/') || s.startsWith('b/')) s = s.slice(2);
  return s;
}

/** The path of a `rename from `/`rename to ` line: the rest of the line, up to the first tab, spaces removed. */
function renamePath(line: string, prefix: string): string {
  let s = line.slice(prefix.length);
  const tab = s.indexOf('\t');
  if (tab >= 0) s = s.slice(0, tab);
  return s.trim();
}

function parseUnified(lines: string[], strict: boolean): Edit[] {
  const sections: Section[] = [];
  let sec: Section | null = null;
  // state: 'none' before any section, 'header' before the section's first hunk, 'body' inside a hunk,
  // 'between' after a hunk's body ended but still in the section.
  let state: 'none' | 'header' | 'body' | 'between' = 'none';
  let hunk: Hunk | null = null;

  // The parser records body lines with their raw form so trailing empty lines can be told from context lines
  // whose text is a single space's remainder. `rawEmpty` marks a patch line that was completely empty.
  type RawBody = BodyLine & { rawEmpty?: boolean };
  let rawBody: RawBody[] = [];

  const flushHunk = (): void => {
    if (hunk !== null) {
      while (rawBody.length > 0 && rawBody[rawBody.length - 1]!.rawEmpty) rawBody.pop();
      hunk.body = rawBody.map((b) => (b.k === 'bs' ? { k: 'bs' as const } : { k: b.k, text: b.text }));
      sec!.hunks.push(hunk);
      hunk = null;
      rawBody = [];
    }
  };

  const startSection = (): void => {
    flushHunk();
    sec = { newFile: false, deletedFile: false, hunks: [] };
    sections.push(sec);
    state = 'header';
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!;
    const no = i + 1;

    if (line.startsWith('diff --git ')) {
      const m = DIFF_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${no}`);
      startSection();
      sec!.diffOld = m[1]!;
      sec!.diffNew = m[2]!;
      continue;
    }

    if (line.startsWith('@@')) {
      const m = HUNK_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${no}`);
      if (sec === null) throw new Exit2(`error: parse ${no}`);
      flushHunk();
      hunk = { a: Number(m[1]), body: [] };
      rawBody = [];
      state = 'body';
      continue;
    }

    const pairAhead = line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ');

    if (state === 'body' && !pairAhead) {
      const c = line[0];
      if (c === ' ' || c === '-' || c === '+' || c === '\\' || line === '') {
        if (c === ' ') rawBody.push({ k: 'ctx', text: line.slice(1) });
        else if (c === '-') rawBody.push({ k: 'del', text: line.slice(1) });
        else if (c === '+') rawBody.push({ k: 'add', text: line.slice(1) });
        else if (c === '\\') rawBody.push({ k: 'bs' });
        else rawBody.push({ k: 'ctx', text: '', rawEmpty: true });
        continue;
      }
      // The body ends at the first other line; fall through and read that line outside the body.
      flushHunk();
      state = 'between';
    }

    if (pairAhead) {
      if (state === 'header' && sec!.minus === undefined) {
        // The pair is this section's own header: it names the section's old and new paths.
        sec!.minus = headerPath(line);
        sec!.plus = headerPath(lines[i + 1]!);
        i += 1;
        continue;
      }
      if (state === 'none' || state === 'between' || state === 'body' || (state === 'header' && sec!.minus !== undefined)) {
        // A ---/+++ pair that is not part of the current section's header starts a new section (and ends a body).
        startSection();
        sec!.minus = headerPath(line);
        sec!.plus = headerPath(lines[i + 1]!);
        i += 1;
        continue;
      }
    }

    if (state === 'header') {
      if (line.startsWith('--- ')) {
        // Inside a section's header, a --- line not followed by a +++ line is a parse error at the next line.
        throw new Exit2(`error: parse ${no + 1}`);
      }
      if (line.startsWith('rename from ')) sec!.renameFrom = renamePath(line, 'rename from ');
      else if (line.startsWith('rename to ')) sec!.renameTo = renamePath(line, 'rename to ');
      else if (line.startsWith('new file mode')) sec!.newFile = true;
      else if (line.startsWith('deleted file mode')) sec!.deletedFile = true;
      else if (line.startsWith('--- ') || line.startsWith('+++ ')) {
        // Unreachable for --- (handled above); a stray +++ line in a header is ignored like any other line.
      }
      // index, mode, similarity and every other header line are ignored.
      continue;
    }

    if (state === 'none' || state === 'between') {
      // Outside bodies, lines that are not part of a section header are ignored (prose, code fences).
      continue;
    }
  }
  flushHunk();

  return sections.map((s) => unifiedEdit(s, strict));
}

/** Apply one section's hunks to a file's text; a hunk failure fails the section with that hunk's code. */
function applyHunks(text: string, hunks: Hunk[], strict: boolean, failPath: string): string {
  const model = splitText(text);
  const work = model.lines.slice();
  let fin = model.final;
  let d = 0;
  for (const h of hunks) {
    const old = bodyOld(h.body);
    const neu = bodyNeu(h.body);
    if (old.length === 0) {
      const adds = neu.map((it) => ('add' in it ? it.add : ''));
      const at = Math.min(h.a + d, work.length);
      const atEnd = at === work.length;
      work.splice(at, 0, ...adds);
      if (atEnd) fin = !noEolAfterLastCtxOrAdd(h.body);
      d += neu.length;
      continue;
    }
    const m = matchOld(work, old, strict, 0);
    if (m === null) throw new Fail('no-match', failPath);
    const target = h.a - 1 + d;
    let pos = m.positions[0]!;
    for (const p of m.positions) {
      if (Math.abs(p - target) < Math.abs(pos - target)) pos = p;
    }
    const repl = renderNeu(work, pos, old, neu, m.tier);
    const runsToEnd = pos + old.length === work.length;
    work.splice(pos, old.length, ...repl);
    if (runsToEnd) fin = !noEolAfterLastCtxOrAdd(h.body);
    d += neu.length - old.length;
  }
  return joinText({ lines: work, br: model.br, final: fin });
}

function unifiedEdit(s: Section, strict: boolean): Edit {
  const oldPath = s.minus ?? s.renameFrom ?? s.diffOld;
  const newPath = s.plus ?? s.renameTo ?? s.diffNew;
  const isCreate = s.minus === '/dev/null' || s.newFile;
  const isDelete = s.plus === '/dev/null' || s.deletedFile;
  return (tree) => {
    if (isCreate) {
      const p = newPath!;
      if (!safePath(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      const addLines: string[] = [];
      let finalBreak = true;
      for (const h of s.hunks) {
        for (const b of h.body) if (b.k === 'add') addLines.push(b.text);
        if (h.body.some((b) => b.k === 'add')) finalBreak = !noEolAfterLastAdd(h.body);
      }
      tree.now.set(p, joinText({ lines: addLines, br: '\n', final: finalBreak }));
      return;
    }
    if (isDelete) {
      const p = oldPath!;
      if (!safePath(p)) throw new Fail('unsafe', p);
      if (tree.get(p) === null) throw new Fail('missing', p);
      tree.now.set(p, null);
      return;
    }
    const from = oldPath!;
    const to = newPath!;
    if (!safePath(from)) throw new Fail('unsafe', from);
    if (to !== from && !safePath(to)) throw new Fail('unsafe', to);
    const before = tree.get(from);
    if (before === null) throw new Fail('missing', from);
    if (to !== from && tree.get(to) !== null) throw new Fail('exists', to);
    const after = applyHunks(before, s.hunks, strict, from);
    if (to !== from) {
      tree.now.set(to, after);
      tree.now.set(from, null);
      tree.moves.push([from, to] as const);
    } else {
      tree.now.set(from, after);
    }
  };
}

// ---------------------------------------------------------------------------
// Section 7: SEARCH/REPLACE blocks.
// ---------------------------------------------------------------------------

interface SrBlock {
  path: string;
  oldLines: string[];
  newLines: string[];
}

function parseSearchReplace(lines: string[], strict: boolean): Edit[] {
  const blocks: SrBlock[] = [];
  let prevEnd = -1; // index of the previous block's >>>>>>> REPLACE line
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i += 1;
      continue;
    }
    const searchNo = i + 1;
    // The path: the nearest line before the SEARCH line, after the previous block's REPLACE line, that is not
    // blank and does not (trimmed) start with three backticks.
    let path: string | null = null;
    for (let j = i - 1; j > prevEnd; j -= 1) {
      const cand = lines[j]!;
      if (isBlank(cand)) continue;
      if (cand.trim().startsWith('```')) continue;
      path = cand.trim();
      break;
    }
    if (path === null) {
      if (prevPath !== null) path = prevPath;
      else throw new Exit2(`error: parse ${searchNo}`);
    }
    // Old lines run to the ======= line, new lines to the >>>>>>> REPLACE line.
    let eq = -1;
    for (let j = i + 1; j < lines.length; j += 1) {
      if (lines[j]!.trimEnd() === '=======') {
        eq = j;
        break;
      }
    }
    if (eq < 0) throw new Exit2(`error: parse ${searchNo}`);
    let rep = -1;
    for (let j = eq + 1; j < lines.length; j += 1) {
      if (lines[j]!.trimEnd() === '>>>>>>> REPLACE') {
        rep = j;
        break;
      }
    }
    if (rep < 0) throw new Exit2(`error: parse ${searchNo}`);
    blocks.push({ path, oldLines: lines.slice(i + 1, eq), newLines: lines.slice(eq + 1, rep) });
    prevPath = path;
    prevEnd = rep;
    i = rep + 1;
  }
  return blocks.map((b) => srEdit(b, strict));
}

function srEdit(b: SrBlock, strict: boolean): Edit {
  return (tree) => {
    const p = b.path;
    if (!safePath(p)) throw new Fail('unsafe', p);
    if (b.oldLines.length === 0) {
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, joinText({ lines: b.newLines.slice(), br: '\n', final: true }));
      return;
    }
    const before = tree.get(p);
    if (before === null) throw new Fail('missing', p);
    const model = splitText(before);
    const m = matchOld(model.lines, b.oldLines, strict, 0);
    if (m === null) throw new Fail('no-match', p);
    if (m.positions.length > 1) throw new Fail('ambiguous', p);
    const pos = m.positions[0]!;
    const neu: NeuLine[] = b.newLines.map((t) => ({ add: t }));
    const repl = renderNeu(model.lines, pos, b.oldLines, neu, m.tier);
    const work = model.lines.slice();
    work.splice(pos, b.oldLines.length, ...repl);
    tree.now.set(p, joinText({ lines: work, br: model.br, final: model.final }));
  };
}

// ---------------------------------------------------------------------------
// Section 8: V4A patches.
// ---------------------------------------------------------------------------

interface V4aChunk {
  anchor: string | null;
  eof: boolean;
  body: BodyLine[];
}

type V4aSection =
  | { kind: 'add'; path: string; fileLines: string[] }
  | { kind: 'delete'; path: string }
  | { kind: 'update'; path: string; moveTo?: string; chunks: V4aChunk[] };

const V4A_ADD = '*** Add File: ';
const V4A_DEL = '*** Delete File: ';
const V4A_UPD = '*** Update File: ';
const V4A_MOVE = '*** Move to: ';
const V4A_EOF = '*** End of File';

function parseV4a(lines: string[], strict: boolean): Edit[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (!isBlank(lines[i]!)) {
      last = i;
      break;
    }
  }
  if (lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);

  const sections: V4aSection[] = [];
  let i = first + 1;
  const isHead = (l: string): boolean => l.startsWith(V4A_ADD) || l.startsWith(V4A_DEL) || l.startsWith(V4A_UPD);

  while (i < last) {
    const line = lines[i]!;
    if (isBlank(line)) {
      i += 1;
      continue;
    }
    if (!isHead(line)) throw new Exit2(`error: parse ${i + 1}`);
    const headNo = i + 1;
    // Collect the section's lines, up to the next section header or the End Patch line; blank lines at the end
    // of a section are not part of it.
    const bodyStart = i + 1;
    let j = bodyStart;
    while (j < last && !isHead(lines[j]!)) j += 1;
    const raw = lines.slice(bodyStart, j);
    while (raw.length > 0 && isBlank(raw[raw.length - 1]!)) raw.pop();
    const at = (k: number): number => bodyStart + k + 1; // 1-based patch line of raw[k]

    if (line.startsWith(V4A_ADD)) {
      const path = line.slice(V4A_ADD.length).trim();
      const fileLines: string[] = [];
      raw.forEach((l, k) => {
        if (l.startsWith('+')) fileLines.push(l.slice(1));
        else if (l === '') fileLines.push('');
        else throw new Exit2(`error: parse ${at(k)}`);
      });
      sections.push({ kind: 'add', path, fileLines });
    } else if (line.startsWith(V4A_DEL)) {
      const path = line.slice(V4A_DEL.length).trim();
      raw.forEach((l, k) => {
        if (!isBlank(l)) throw new Exit2(`error: parse ${at(k)}`);
      });
      sections.push({ kind: 'delete', path });
    } else {
      const path = line.slice(V4A_UPD.length).trim();
      let moveTo: string | undefined;
      let k = 0;
      if (raw.length > 0 && raw[0]!.startsWith(V4A_MOVE)) {
        moveTo = raw[0]!.slice(V4A_MOVE.length).trim();
        k = 1;
      }
      const chunks: V4aChunk[] = [];
      let chunk: V4aChunk | null = null;
      for (; k < raw.length; k += 1) {
        const l = raw[k]!;
        if (l.startsWith('@@')) {
          chunk = { anchor: l.slice(2).trim() === '' ? null : l.slice(2).trim(), eof: false, body: [] };
          chunks.push(chunk);
        } else if (l.trim() === V4A_EOF) {
          if (chunk === null) throw new Exit2(`error: parse ${at(k)}`);
          chunk.eof = true;
          chunk = null;
        } else if (l.startsWith('***')) {
          throw new Exit2(`error: parse ${at(k)}`);
        } else if (l === '' || l[0] === ' ' || l[0] === '-' || l[0] === '+') {
          if (chunk === null) {
            chunk = { anchor: null, eof: false, body: [] };
            chunks.push(chunk);
          }
          if (l === '') chunk.body.push({ k: 'ctx', text: '' });
          else if (l[0] === ' ') chunk.body.push({ k: 'ctx', text: l.slice(1) });
          else if (l[0] === '-') chunk.body.push({ k: 'del', text: l.slice(1) });
          else chunk.body.push({ k: 'add', text: l.slice(1) });
        } else {
          throw new Exit2(`error: parse ${at(k)}`);
        }
      }
      if (chunks.length === 0 && moveTo === undefined) throw new Exit2(`error: parse ${headNo}`);
      sections.push({ kind: 'update', path, moveTo, chunks });
    }
    i = j;
  }
  return sections.map((s) => v4aEdit(s, strict));
}

function v4aEdit(s: V4aSection, strict: boolean): Edit {
  return (tree) => {
    if (s.kind === 'add') {
      if (!safePath(s.path)) throw new Fail('unsafe', s.path);
      if (tree.get(s.path) !== null) throw new Fail('exists', s.path);
      tree.now.set(s.path, joinText({ lines: s.fileLines.slice(), br: '\n', final: true }));
      return;
    }
    if (s.kind === 'delete') {
      if (!safePath(s.path)) throw new Fail('unsafe', s.path);
      if (tree.get(s.path) === null) throw new Fail('missing', s.path);
      tree.now.set(s.path, null);
      return;
    }
    // update
    if (!safePath(s.path)) throw new Fail('unsafe', s.path);
    if (s.moveTo !== undefined && !safePath(s.moveTo)) throw new Fail('unsafe', s.moveTo);
    const before = tree.get(s.path);
    if (before === null) throw new Fail('missing', s.path);
    if (s.moveTo !== undefined && tree.get(s.moveTo) !== null) throw new Fail('exists', s.moveTo);
    const model = splitText(before);
    const work = model.lines.slice();
    let cursor = 0;
    for (const chunk of s.chunks) {
      if (chunk.anchor !== null) {
        const at = matchAnchor(work, chunk.anchor, strict, cursor);
        if (at < 0) throw new Fail('no-match', s.path);
        cursor = at + 1;
      }
      const old = bodyOld(chunk.body);
      const neu = bodyNeu(chunk.body);
      if (old.length === 0) {
        const adds = neu.map((it) => ('add' in it ? it.add : ''));
        const at = chunk.eof ? work.length : cursor;
        work.splice(at, 0, ...adds);
        cursor = at + adds.length;
        continue;
      }
      const m = matchOld(work, old, strict, cursor, chunk.eof);
      if (m === null) throw new Fail('no-match', s.path);
      const pos = m.positions[0]!;
      const repl = renderNeu(work, pos, old, neu, m.tier);
      work.splice(pos, old.length, ...repl);
      cursor = pos + repl.length;
    }
    const after = joinText({ lines: work, br: model.br, final: model.final });
    if (s.moveTo !== undefined) {
      tree.now.set(s.moveTo, after);
      tree.now.set(s.path, null);
      tree.moves.push([s.path, s.moveTo] as const);
    } else {
      tree.now.set(s.path, after);
    }
  };
}

// ---------------------------------------------------------------------------
// Section 9: JSON edits.
// ---------------------------------------------------------------------------

/** One JSON edit object, or null when it is not one (a wrong command, a missing or mistyped key). */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;

  const readExisting = (tree: Tree): string => {
    if (!safePath(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };

  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safePath(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text); // a JSON create writes its file_text byte for byte
    };
  }

  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = readExisting(tree);
      // In a CRLF file the text and both strings are read with every \r\n as \n, and written back with \r\n.
      const model = splitText(before);
      const crlf = model.br === '\r\n';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }

  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = readExisting(tree);
      const model = splitText(before);
      if (line < 0 || line > model.lines.length) throw new Fail('no-match', p);
      const work = model.lines.slice();
      work.splice(line, 0, ...add);
      // The file's final break stays as it was.
      tree.now.set(p, joinText({ lines: work, br: model.br, final: model.final }));
    };
  }

  return null;
}

function parseJson(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trimStart().startsWith('[')) {
    // Array form: any fault is a parse error at the line where the array starts.
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    const edits: Edit[] = [];
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  // JSON Lines form: one edit object a line, blank lines ignored.
  const edits: Edit[] = [];
  lines.forEach((l, i) => {
    if (isBlank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

// ---------------------------------------------------------------------------
// Section 4: detecting the format.
// ---------------------------------------------------------------------------

function detectFormat(lines: string[]): string | null {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// ---------------------------------------------------------------------------
// Sections 1, 2 and 10: the command line, the run, the all-or-nothing write.
// ---------------------------------------------------------------------------

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMAT_NAMES.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };

  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detectFormat(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };

  let edits: Edit[];
  try {
    if (kind === 'json') edits = parseJson(text, lines);
    else if (kind === 'unified') edits = parseUnified(lines, strict);
    else if (kind === 'search-replace') edits = parseSearchReplace(lines, strict);
    else edits = parseV4a(lines, strict);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }

  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    const moveCount = tree.moves.length;
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      tree.moves.length = moveCount;
      fails.push(e);
    }
  }
  if (fails.length > 0) {
    return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  }

  // The changed paths, in byte order of the paths (UTF-8); a qualifying move replaces its D and A lines with
  // one R line that sorts by the new path (section 2).
  const bytewise = (a: string, b: string): number => Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
  const moveByNew = new Map<string, string>();
  for (const [from, to] of tree.moves) {
    if (tree.original(from) !== null && tree.final(from) === null && tree.original(to) === null && tree.final(to) !== null) {
      moveByNew.set(to, from);
    }
  }
  const moveOlds = new Set(moveByNew.values());
  const rows: Array<{ key: string; line: string }> = [];
  for (const p of tree.now.keys()) {
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    if (moveByNew.has(p)) {
      rows.push({ key: p, line: `R ${moveByNew.get(p)!} -> ${p}` });
    } else if (moveOlds.has(p)) {
      continue; // the old path of a qualifying move: its D line is replaced by the R line
    } else {
      rows.push({ key: p, line: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
    }
  }
  rows.sort((x, y) => bytewise(x.key, y.key));

  if (!check) {
    for (const p of [...tree.now.keys()].sort(bytewise)) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  const n = rows.length;
  return { code: 0, out: [...rows.map((r) => r.line), `ok: ${n} ${n === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
