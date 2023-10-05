pdf: compileplan compilepaper
	xdg-open *.pdf

compileplan:
	pdflatex -shell-escape -pdf project-plan.tex

compilepaper:
	pdflatex -shell-escape project-paper
	bibtex project-paper
	pdflatex project-paper
	pdflatex project-paper

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.out
	rm -f *.bbl
	rm -f *.blg
	rm -f *.bcf
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.toc
	rm -f *.run.xml
	rm -f *.dvi
	rm -f *.synctex.gz
