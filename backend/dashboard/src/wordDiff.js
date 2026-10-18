// Word-by-word difference between two texts (longest common subsequence),
// used to highlight what each proposal changes in the Javadoc.
// Returns [{ text, kind }] with kind "same" | "add" | "del".

function tokens(text) {
  return (text || "").split(/(\s+)/).filter((t) => t !== "");
}

export function wordDiff(before, after) {
  const a = tokens(before);
  const b = tokens(after);
  // Large inputs: fall back to a line-free, whole-text comparison (keeps the UI responsive).
  if (a.length * b.length > 400000) {
    return before === after ? [{ text: after, kind: "same" }] : [{ text: before, kind: "del" }, { text: after, kind: "add" }];
  }
  const dp = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = [];
  const push = (text, kind) => {
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ text, kind });
  };
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      push(a[i], "same");
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      push(a[i], "del");
      i += 1;
    } else {
      push(b[j], "add");
      j += 1;
    }
  }
  while (i < a.length) push(a[i++], "del");
  while (j < b.length) push(b[j++], "add");
  return out;
}
