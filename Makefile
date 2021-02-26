all: build/data-structures/fact-timeseries.pdf build/data-structures/lst_images.pdf


build/data-structures/fact-timeseries.pdf: data-structures/fact-timeseries.py | build/data-structures
	MAPLOTLIBRC=matplotlibrc TEXINPUTS=$$(pwd): python $< $@

build/data-structures/lst_images.pdf: data-structures/lst_images.py | build/data-structures
	MAPLOTLIBRC=matplotlibrc TEXINPUTS=$$(pwd): python $< $@


build/data-structures:
	mkdir -p $@
