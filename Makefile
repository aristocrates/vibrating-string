SET  = set3
OPEN = xdg-open

$(SET).pdf: $(SET).tex Makefile integrator.py problem.py physicalstring.py \
            testsuite.py stringtest.py
	pdflatex -shell-escape $(SET).tex
	pdflatex -shell-escape $(SET).tex

.PHONY: clean view gifs

animated_gif/%.gif:
	echo $(subst animated_gif/,,$(subst .gif,,$@))
	sh mkgif.sh $(subst animated_gif/,,$(subst .gif,,$@))

gifs: animated_gif/plots_fixed.gif animated_gif/plots_sinusoid.gif animated_gif/plots_fixed_damped.gif animated_gif/plots_fixed_ergo.gif animated_gif/plots_fixed_nonergo.gif animated_gif/plots_fixed_anderson.gif

view: $(SET).pdf
	$(OPEN) $(SET).pdf

clean:
	rm -f $(SET).pdf $(SET).log $(SET).aux
