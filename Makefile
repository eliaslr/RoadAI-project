pdf: compiletex
	xdg-open *.pdf

compiletex:
	pdflatex -shell-escape -pdf project-plan.tex

clean:
	rm *.aux
	rm *.log
	rm *.out
	rm *.pdf
