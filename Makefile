pdf: compiletex
	open *.pdf

compiletex:
	pdflatex -shell-escape -pdf *.tex

clean:
	rm *.aux
	rm *.log
	rm *.out
	rm article-summaries.pdf
	rm project-plan.pdf
