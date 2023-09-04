pdf: compiletex
	open *.pdf
	
compiletex:
	pdflatex -shell-escape -pdf *.tex
