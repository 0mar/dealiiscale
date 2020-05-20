ell-par.pdf: ell-par.tex plot*.pdf
	pdflatex ell-par.tex && \
		pdflatex ell-par.tex

plot*.pdf: plot_convergence.py *-convergence.txt
	python3 plot_convergence.py

plot_convergence.py: plot_convergence.ipynb
	jupyter nbconvert --to script plot_convergence.ipynb

%::
	-echo "Make can't find $$(pwd)$@!!!"
