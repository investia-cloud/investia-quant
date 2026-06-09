# investia-quant — Release 2026.1

**Data**: 2026-06-09
**Commit**: 19d4b63
**Branch**: refactor/libs-py

## Deploy su VPS

```bash
# 1. Copia la release sulla VPS
rsync -av releases/2026.1/ tslab.investia.cloud:~/investia-quant/releases/2026.1/

# 2. Sulla VPS: installa l'ambiente
ssh tslab.investia.cloud "~/investia-quant/releases/2026.1/scripts/install.sh"

# 3. Sulla VPS: aggiorna il symlink current
ssh tslab.investia.cloud "ln -sfn ~/investia-quant/releases/2026.1 ~/investia-quant/releases/current"

# 4. Verifica
ssh tslab.investia.cloud "~/investia-quant/releases/current/.venv/bin/iq --help"
```

## Rollback

```bash
# Torna alla release precedente
ssh tslab.investia.cloud "ln -sfn ~/investia-quant/releases/2026.0 ~/investia-quant/releases/current"
```

## Struttura

```
2026.1/
├── lib/                  librerie runtime
├── investia_quant/       CLI iq
├── inputs/               dati WFO + config portafogli
├── cache/                cache ticker/ISIN
├── outputs/              output report (vuota all'installazione)
├── scripts/
│   ├── portfolios.conf   configurazione portafogli
│   ├── install.sh        setup venv
│   └── crontab.txt       entries cron
└── .venv/                venv (creato da install.sh, non in git)
```
