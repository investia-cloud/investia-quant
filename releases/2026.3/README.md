# investia-quant — Release 2026.3

**Data**: 2026-06-09
**Commit**: e0717b5
**Branch**: refactor/libs-py

## Deploy su VPS

```bash
./scripts/deploy.sh 2026.3
```

## Rollback

```bash
ssh tslab.investia.cloud "ln -sfn ~/investia-quant/releases/<versione_precedente> ~/investia-quant/releases/current"
```

## Struttura

```
2026.3/
├── lib/                  librerie runtime
├── investia_quant/       CLI iq
├── inputs/               dati WFO
├── cache/                cache ticker/ISIN
├── outputs/              output report (vuota)
├── scripts/
│   ├── portfolios.conf
│   ├── install.sh        setup venv + .pth per lib/
│   └── crontab.txt
└── .venv/                venv (non in git)
```
