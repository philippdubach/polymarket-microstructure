#!/usr/bin/env bash
# Build the arXiv submission bundle.
#
# Output: paper/arxiv/anatomy_arxiv.tar.gz containing:
#   - anatomy.tex (flattened: all \input{} resolved inline)
#   - anatomy.bbl (bundled so arXiv does not run bibtex)
#   - figures/*.png (referenced by the flattened tex)
#   - tables/*.tex (referenced by the flattened tex; latexpand inlines table bodies but keeping the directory keeps the bundle inspectable)
#
# arXiv expects to run pdflatex twice on the flattened tex; the bundled
# .bbl satisfies the bibliography step without bibtex.
#
# Usage: bash scripts/build_arxiv_bundle.sh
# Requires: latexpand (ships with TeX Live)

set -euo pipefail

cd "$(dirname "$0")/.."

PAPER_DIR=paper
BUILD_DIR=$PAPER_DIR/build
OUT_DIR=$PAPER_DIR/arxiv
BUNDLE_DIR=$OUT_DIR/anatomy_arxiv

# Re-run a clean compile to make sure .bbl is current
echo "==> Compiling anatomy.tex with bibtex"
( cd "$PAPER_DIR" && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex >/dev/null 2>&1 \
  && bibtex build/anatomy >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex >/dev/null 2>&1 )

if [ ! -f "$BUILD_DIR/anatomy.bbl" ]; then
  echo "ERROR: $BUILD_DIR/anatomy.bbl missing after compile"
  exit 1
fi

# Reset bundle directory
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/figures"

# Flatten anatomy.tex: inline all \input{}
echo "==> Flattening with latexpand"
( cd "$PAPER_DIR" && latexpand --empty-comments anatomy.tex ) > "$BUNDLE_DIR/anatomy.tex"

# Bundle the bib output
cp "$BUILD_DIR/anatomy.bbl" "$BUNDLE_DIR/anatomy.bbl"

# Copy figures referenced by the flattened tex
cp -r "$PAPER_DIR/figures/." "$BUNDLE_DIR/figures/"

# Sanity: confirm no \input remains and at least one \begin{document}
if grep -q "\\\\input{" "$BUNDLE_DIR/anatomy.tex"; then
  echo "WARNING: \\input{} survived flattening"
fi
if ! grep -q "begin{document}" "$BUNDLE_DIR/anatomy.tex"; then
  echo "ERROR: flattened tex missing \\begin{document}"
  exit 1
fi

# Verify the bundle compiles standalone
echo "==> Sanity-compiling bundled tex"
( cd "$BUNDLE_DIR" && pdflatex -interaction=nonstopmode anatomy.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode anatomy.tex >/dev/null 2>&1 )

if [ ! -f "$BUNDLE_DIR/anatomy.pdf" ]; then
  echo "ERROR: bundled tex did not produce a PDF"
  exit 1
fi

PAGES=$(pdfinfo "$BUNDLE_DIR/anatomy.pdf" | grep '^Pages:' | awk '{print $2}')
echo "  bundled PDF: $PAGES pages"

# Strip the build artefacts the bundle does not need to ship
rm -f "$BUNDLE_DIR/anatomy.aux" "$BUNDLE_DIR/anatomy.log" \
      "$BUNDLE_DIR/anatomy.out" "$BUNDLE_DIR/anatomy.toc" \
      "$BUNDLE_DIR/anatomy.blg" "$BUNDLE_DIR/anatomy.pdf"

# Make the tarball arXiv expects
TARBALL="$OUT_DIR/anatomy_arxiv.tar.gz"
echo "==> Creating $TARBALL"
( cd "$OUT_DIR" && tar -czf anatomy_arxiv.tar.gz anatomy_arxiv )

ls -la "$TARBALL"
echo "==> Bundle contents:"
tar -tzf "$TARBALL" | head -20

echo ""
echo "Bundle ready. Upload $TARBALL to arXiv via the new-submission form."
