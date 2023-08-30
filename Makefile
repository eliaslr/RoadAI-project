pdf: compiletex
	open *.pdf
	cat Did you know that pdf stands for portable document format? Because I didn't...
	
compiletex:
	pdflatex -shell-escape *.tex
