# Requisition — design theme

The interface borrows from the one document every clinician already knows how to
read: the **lab requisition and result slip**. Not hospital-brand blue, not a
generic chat window. Pale specimen paper, hairline rules, monospaced values in
tabular columns, and small capitalised field labels — the vernacular of a printed
result sheet, rebuilt as software.

The reason is functional, not decorative. A result slip is designed to be scanned
under time pressure: label first, then values in a fixed column, then flags. That
is exactly the reading order this app needs.

## Palette

Light, cool, and low-chroma so the only saturated things on screen are the ones
that carry meaning — the active patient and a lab flag.

| Token | Hex | Role |
| --- | --- | --- |
| `--paper` | `#F6F8F9` | Page background. Cool grey-green, the colour of carbonless lab paper. |
| `--surface` | `#FFFFFF` | Cards, message surfaces, the composer. |
| `--ink` | `#16262F` | Primary text. Deep slate, never pure black — print ink on stock, not screen black. |
| `--slate` | `#64808E` | Secondary text, field labels, timestamps. |
| `--rule` | `#DCE4E8` | Hairlines. Does the work borders and shadows would otherwise do. |
| `--specimen` | `#0F766E` | The single action colour: selected patient, send, focus rings. |
| `--flag-high` | `#A85B00` | Amber. Mirrors the `H` flag on a result slip. |
| `--flag-low` | `#1D5FA8` | Blue. Mirrors the `L` flag. |

Deliberately excluded: red. On a clinical surface red reads as *critical value*.
Nothing in this app is a clinical judgement, so nothing earns red — errors use
slate with an amber rule instead.

## Type

Two roles, both from system stacks so the app has no network dependency and looks
native on the machine it runs on.

- **Interface** — `system-ui` stack. Sentence case, comfortable measure, for
  prose answers and controls.
- **Data** — monospace stack (`ui-monospace`, `SF Mono`, `Cascadia Mono`, Menlo).
  Every identifier, numeric value, unit, date and LOINC code. All numerals use
  `font-variant-numeric: tabular-nums` so digits align in a column the way they
  do on a printed panel.
- **Field labels** — monospace, 11px, uppercase, `0.09em` letter-spacing. The
  small stamped captions of a paper form (`PATIENT`, `RECORD SPAN`, `SPECIMEN`).

Type does the structural work; there are no icons.

## Layout

A guided two-step opening, then it gets out of the way.

```
STEP 1 — select                    STEP 2 — converse
┌───────────────────────────┐      ┌───────────────────────────┐
│ Patient History Navigator │      │ ⌐ SPECIMEN ─────────  new │ ← sticky strip
│ Select a record to open   │      │ ON-SYN-73B97EDEEB         │
│                           │      ├───────────────────────────┤
│ ┌────────┐ ┌────────┐     │      │  you ▸ what labs…         │
│ │ label  │ │ label  │ …   │  →   │  ┌─ answer ─────────────┐ │
│ │ 44 doc │ │ 54 doc │     │      │  │ prose + [Evidence 1] │ │
│ └────────┘ └────────┘     │      │  └─ evidence (8) ▾ ────┘ │
│                           │      ├───────────────────────────┤
│ synthetic data notice     │      │ [ ask about this record ] │
└───────────────────────────┘      └───────────────────────────┘
```

Selection is a full-screen decision, because choosing the wrong patient
invalidates everything after it. Once made, it compresses to a one-line specimen
strip and never blocks the conversation again. Restarting is always one click in
that strip.

The conversation is a single column with no sidebar. Evidence is collapsed under
each answer rather than parked in a panel, because it belongs to that specific
claim.

## Signature: the result slip

Tool output is not rendered as chat prose. When the agent runs a deterministic
tool, its result is drawn as **a lab result slip** — the analyte name, a
right-aligned tabular value, its unit, and a flag column. A numeric trend gets an
inline sparkline with its first and last values labelled, sized to sit on the
text baseline like a printed mini-chart.

This is the one place the design spends boldness. It also encodes something true:
these tools return *computed, inspectable* values, not generated text, so they
should not look like generated text.

## Restraint

- No shadows except a 1px hairline and one soft lift on the active card.
- No gradients, no glass, no rounded-pill buttons; radius is 3px, the corner of a
  cut paper form.
- Motion is limited to a 120ms opacity/translate on new messages and the
  sparkline draw. `prefers-reduced-motion` removes both.
- The synthetic-data notice is permanent and quiet — a stamped caption, never a
  dismissible banner, because it must be true on every screenshot.

## Voice

Field labels name what a clinician controls: `PATIENT`, `RECORD SPAN`,
`MEASUREMENTS`. Actions say what they do — "Open record", "New session", "Ask".
The empty state is an instruction, not a greeting. Errors state what happened and
the next move: "The endpoint is waking up. Try again in a moment." Nothing
apologises and nothing is vague.
