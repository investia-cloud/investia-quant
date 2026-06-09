# investia-quant — Release 2026.1

**Data**: 2026-06-09
**Commit**: c0c6f06
**Branch**: refactor/libs-py

## Deploy su VPS

```bash
./scripts/deploy.sh 2026.1 <vps_host> <install_dir>
# Es: ./scripts/deploy.sh 2026.1 tslab.investia.cloud /home/luca
```

## Rollback

```bash
ssh <vps_host> "ln -sfn <install_dir>/investia-quant/releases/<versione_precedente> <install_dir>/investia-quant/releases/current"
```

## Struttura

```
2026.1/
├── lib/                  librerie runtime (.py)
├── investia_quant/       CLI iq
├── inputs/               dati WFO
├── cache/                cache ticker/ISIN
├── outputs/              output report (vuota)
├── logs/                 log cron (vuota)
├── scripts/
│   ├── portfolios.conf
│   ├── install.sh
│   └── crontab.txt
└── .venv/                venv (creato da install.sh, non in git)
```
