report.pdf: report.tex plot*.pdf
	pdflatex report.tex && \
		pdflatex report.tex

plot*.pdf: plot_convergence.py *-convergence.txt
	python3 plot_convergence.py

plot_convergence.py: plot_convergence.ipynb
	jupyter nbconvert --to script plot_convergence.ipynb

%::
	-echo "Make can't find $$(pwd)$@!!!"
