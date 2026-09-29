# Tables and figures

`data/` holds the numbers of the tables and figures, `paper/` the figures of the paper.

```bash
python analysis/make_tables.py      # Tables 1, 2, 9, 10
python analysis/plot_fig1.py        # -> analysis/out/fig1.pdf; same for fig3a, fig3b, fig3c, fig4, fig5
```

`collect.py` computes scores and the average normalized score from evaluation outputs.
