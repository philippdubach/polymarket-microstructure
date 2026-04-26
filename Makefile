PAPER := paper/anatomy
BUILDDIR := paper/build

.PHONY: pdf assets clean

pdf: assets
	@mkdir -p $(BUILDDIR)
	cd paper && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex
	cd paper && bibtex build/anatomy || true
	cd paper && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex
	cd paper && pdflatex -interaction=nonstopmode -output-directory=build anatomy.tex
	@echo "Wrote $(BUILDDIR)/anatomy.pdf"

assets:
	uv run python scripts/build_paper_assets.py || true

clean:
	rm -rf $(BUILDDIR)
