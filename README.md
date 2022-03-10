 book2-plots

Repository to gather plot / figure generation code for the ML and Physics book


## Plots using python

The needed packages are defined in `environment.yml`, install anaconda and
create the environment using

```
conda env create -f environment.yml
```

Activate the environment:

```
$ conda activate book-plots
```


## Create the plots

Simply run make

```
$ make
```


## Adding more plots

When adding new plots, make sure they are built by the `Makefile` and
all requirements needed are added to the `environment.yml`.

If you need input data, either add the files to the repository (if small enough)
or download them from somewhere using the `Makefile`.

Make sure plots are stored in `build/section-xy/plot.pdf`.

Make sure to create vector graphics (best would be pdf).
