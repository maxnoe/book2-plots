FACT_URL = https://factdata.app.tu-dortmund.de/fp2


all: build/data-structures/fact-timeseries.pdf build/data-structures/lst_images.pdf
all: $(addprefix build/inverse-problems/, fact_unfolding.pdf fact_response.pdf tau_vs_correlation.pdf)


build/data-structures/fact-timeseries.pdf: data-structures/fact-timeseries.py | build/data-structures
	MAPLOTLIBRC=matplotlibrc TEXINPUTS=$$(pwd): python $< $@

build/data-structures/lst_images.pdf: data-structures/lst_images.py | build/data-structures
	MAPLOTLIBRC=matplotlibrc TEXINPUTS=$$(pwd): python $<



build/inverse-problems/fact_response.pdf: build/inverse-problems/fact_unfolding.pdf
build/inverse-problems/tau_vs_correlation.pdf: build/inverse-problems/fact_unfolding.pdf

build/inverse-problems/fact_unfolding.pdf: build/inverse-problems/open_crab_sample_dl3.hdf5
build/inverse-problems/fact_unfolding.pdf: build/inverse-problems/gamma_test_dl3.hdf5
build/inverse-problems/fact_unfolding.pdf: build/inverse-problems/gamma_corsika_headers.hdf5
build/inverse-problems/fact_unfolding.pdf: matplotlibrc
build/inverse-problems/fact_unfolding.pdf: inverse-problems/open_crab_sample_unfolding.py
	MAPLOTLIBRC=matplotlibrc TEXINPUTS=$$(pwd): python $<


build/inverse-problems/%.hdf5: | build/inverse-problems
	curl --fail -Lo $@ $(FACT_URL)/$*.hdf5

build/inverse-problems:
	mkdir -p $@

build/data-structures:
	mkdir -p $@
