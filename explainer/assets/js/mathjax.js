// MathJax config for MkDocs Material + pymdownx.arithmatex (generic mode).
//
// pymdownx.arithmatex with generic:true wraps math in elements with
// class "arithmatex" containing escaped \(...\) or \[...\] delimiters.
// We tell MathJax to:
//   (1) only process elements with class "arithmatex"
//   (2) recognize the escaped \(..\) inline and \[..\] display delimiters
//   (3) re-typeset after MkDocs Material's instant-navigation page swaps

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// MkDocs Material exposes a `document$` Observable for instant navigation.
// Re-typeset math whenever a new page is swapped in, otherwise math from
// the previous page persists and the new page's math doesn't render.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
    }
  });
}
