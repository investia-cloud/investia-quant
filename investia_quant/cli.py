"""
investia_quant/cli.py — CLI `iq`
Entry point: iq run / iq report / iq analyze
"""

import sys
import os
import datetime
import click

# ---------------------------------------------------------------------------
# Default start date per iq report
# R-portfolio: remoto — r_functions aggiusta automaticamente alla prima selezione disponibile
# K-portfolio: YTD — performance dell'anno corrente
# ---------------------------------------------------------------------------
_DEFAULT_ANALYSIS_START_R = "2015-01-01"
_DEFAULT_ANALYSIS_START_K = f"{datetime.date.today().year}-01-01"


def _setup_libs_path():
    """Aggiunge notebooks/libs_py al sys.path se non già presente."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libs_py = os.path.join(root, "notebooks", "libs_py")
    if libs_py not in sys.path:
        sys.path.insert(0, libs_py)


def _load_all_libs():
    """Importa tutte le librerie runtime e restituisce il namespace."""
    _setup_libs_path()
    # Path assoluti dal root progetto via env vars (letti da u_functions)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("IQ_INPUTS_DIR",  os.path.join(root, "inputs"))
    os.environ.setdefault("IQ_OUTPUTS_DIR", os.path.join(root, "outputs"))
    os.environ.setdefault("IQ_CACHE_DIR",   os.path.join(root, "cache"))

    import importlib
    ns = {}
    # Ordine critico: k_tickers prima di k_portfolios e r_portfolios
    libs = [
        "u_functions",
        "k_tickers",       # deve precedere k_portfolios e r_portfolios
        "r_functions",
        "k_functions",
        "s_functions",
        "t_functions",
        "mc_functions",
        "k_strategies",
        "k_portfolios",    # dipende da k_tickers
        "r_portfolios",    # dipende da k_tickers
        "l_portfolios",
    ]
    for lib in libs:
        try:
            mod = importlib.import_module(lib)
            # Inietta il namespace accumulato nel modulo così le dipendenze sono visibili
            for k, v in ns.items():
                if not hasattr(mod, k):
                    try:
                        setattr(mod, k, v)
                    except Exception:
                        pass
            ns.update({k: v for k, v in vars(mod).items() if not k.startswith("__")})
        except Exception as e:
            click.echo(f"[WARN] Impossibile importare {lib}: {e}", err=True)

    return ns


def _resolve_portfolio(ptf_name: str, ns: dict):
    """
    Risolve --ptf <nome> cercando in R_PORTFOLIO_REGISTRY e K_PORTFOLIO_REGISTRY.
    Ritorna (portfolio_obj, kind) dove kind è 'R' o 'K'.
    """
    r_registry = ns.get("R_PORTFOLIO_REGISTRY", {})
    k_registry = ns.get("K_PORTFOLIO_REGISTRY", {})

    if ptf_name in r_registry:
        return r_registry[ptf_name], "R"
    if ptf_name in k_registry:
        return k_registry[ptf_name], "K"

    # Fallback: cerca direttamente per nome variabile nel namespace
    obj = ns.get(ptf_name)
    if obj is not None and isinstance(obj, dict):
        if "trading_systems" in obj:
            return obj, "K"
        if "tickers" in obj:
            return obj, "R"

    available_r = list(r_registry.keys())
    available_k = list(k_registry.keys())
    raise click.ClickException(
        f"Portafoglio '{ptf_name}' non trovato.\n"
        f"  R-portfolio disponibili: {available_r}\n"
        f"  K-portfolio disponibili: {available_k}"
    )


def _get_credentials(ns):
    load_email_credentials = ns.get("load_email_credentials")
    if load_email_credentials is None:
        raise click.ClickException("load_email_credentials non trovata in u_functions.")
    return load_email_credentials()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def app():
    """investia-quant CLI — gestione portafogli quantitativi."""
    pass


# ---------------------------------------------------------------------------
# iq run
# ---------------------------------------------------------------------------

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio (es. alpha_world, us_trading_2026)")
@click.option("--recipient", default=None, help="Email destinatario")
@click.option("--report-date", default=None, help="Data fine report YYYY-MM-DD (default: oggi)")
@click.option("--dry-run", is_flag=True, default=False, help="Simula senza inviare email")
@click.option("--verbose", is_flag=True, default=False, help="Output verboso")
@click.option("--no-send", is_flag=True, default=False, help="Non inviare email (solo esegui)")
@click.option("--wfo-results-dir", default=None, help="Override directory risultati WFO")
def run(ptf, recipient, report_date, dry_run, verbose, no_send, wfo_results_dir):
    """Esecuzione runtime: genera e invia report segnali (R e K portfolio)."""
    click.echo("[iq run] Caricamento librerie...")
    ns = _load_all_libs()

    click.echo(f"[iq run] Risolvo portafoglio: {ptf}")
    portfolio_obj, kind = _resolve_portfolio(ptf, ns)
    click.echo(f"[iq run] Tipo: {kind}-portfolio — {portfolio_obj.get('Title', ptf)}")

    send_report = not no_send and not dry_run

    if kind == "R":
        r_run_portfolio = ns.get("r_run_portfolio")
        if r_run_portfolio is None:
            raise click.ClickException("r_run_portfolio non trovata in r_functions.")
        wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS")
        if send_report:
            if recipient is None:
                raise click.ClickException("Destinatario non specificato. Usa --recipient <email>.")
            sender_email, sender_password = _get_credentials(ns)
            click.echo(f"[iq run] Eseguo r_run_portfolio → {recipient}")
            out = r_run_portfolio(
                portfolio=portfolio_obj,
                report_end_date=report_date,
                year=None,
                wfo_results_dir=wfo_dir,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_email=recipient,
                subject=None,
                verbose=verbose,
                dry_run=dry_run,
                debug=verbose,
            )
            click.echo(f"[iq run] Completato. Output: {out}")
        else:
            click.echo("[iq run] --no-send attivo: r_run_portfolio NON eseguito.")

    elif kind == "K":
        k_run_portfolio = ns.get("k_run_portfolio")
        if k_run_portfolio is None:
            raise click.ClickException("k_run_portfolio non trovata in k_functions.")
        wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_T_WFO_RESULTS_DIR", "../../inputs/WFO_T_RUN_RESULTS")
        if send_report:
            if recipient is None:
                raise click.ClickException("Destinatario non specificato. Usa --recipient <email>.")
            sender_email, sender_password = _get_credentials(ns)
            click.echo(f"[iq run] Eseguo k_run_portfolio → {recipient}")
            out = k_run_portfolio(
                portfolio_cfg=portfolio_obj,
                report_end_date=report_date,
                verbose=verbose,
                wfo_results_dir=wfo_dir,
                create_structure=False,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_email=recipient,
                subject=None,
                check_open_trades=False,
                check_close_trades=False,
                generate_charts=True,
                max_attachments_mb=15,
                max_attachments_count=10,
                attach_mode="signals_only",
            )
            click.echo(f"[iq run] Completato. Output: {out}")
        else:
            click.echo("[iq run] --no-send attivo: k_run_portfolio NON eseguito.")


# ---------------------------------------------------------------------------
# iq report
# ---------------------------------------------------------------------------

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio")
@click.option("--recipient", default=None, help="Email destinatario")
@click.option("--start-date", default=None,
              help="Data inizio analisi YYYY-MM-DD (default: 2015-01-01 per R, YTD per K)")
@click.option("--end-date", default=None, help="Data fine analisi YYYY-MM-DD")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--no-send", is_flag=True, default=False, help="Non inviare email")
@click.option("--wfo-results-dir", default=None)
def report(ptf, recipient, start_date, end_date, verbose, no_send, wfo_results_dir):
    """Genera e invia report performance (R e K portfolio)."""
    click.echo("[iq report] Caricamento librerie...")
    ns = _load_all_libs()

    click.echo(f"[iq report] Risolvo portafoglio: {ptf}")
    portfolio_obj, kind = _resolve_portfolio(ptf, ns)
    click.echo(f"[iq report] Tipo: {kind}-portfolio — {portfolio_obj.get('Title', ptf)}")

    send = not no_send

    if kind == "R":
        # Default: remoto — r_functions aggiusta alla prima selezione disponibile
        effective_start = start_date if start_date is not None else _DEFAULT_ANALYSIS_START_R

        run_rotational_portfolio_performance = ns.get("run_rotational_portfolio_performance")
        if run_rotational_portfolio_performance is None:
            raise click.ClickException("run_rotational_portfolio_performance non trovata in r_functions.")
        wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS")
        if send:
            if recipient is None:
                raise click.ClickException("Destinatario non specificato. Usa --recipient <email>.")
            sender_email, sender_password = _get_credentials(ns)
            click.echo(f"[iq report] Eseguo run_rotational_portfolio_performance → {recipient} (start={effective_start})")
            out = run_rotational_portfolio_performance(
                portfolio=portfolio_obj,
                analisys_start_date=effective_start,
                analisys_end_date=end_date,
                wfo_results_dir=wfo_dir,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_email=recipient,
                show_report=verbose,
                debug=verbose,
                verbose=verbose,
                auto_adjust=True,
            )
            click.echo("[iq report] Completato.")
        else:
            click.echo("[iq report] --no-send attivo.")

    elif kind == "K":
        # Default: YTD (inizio anno corrente)
        effective_start = start_date if start_date is not None else _DEFAULT_ANALYSIS_START_K

        run_ts_portfolio_performance = ns.get("run_ts_portfolio_performance")
        if run_ts_portfolio_performance is None:
            raise click.ClickException("run_ts_portfolio_performance non trovata in k_functions.")
        wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_T_WFO_RESULTS_DIR", "../../inputs/WFO_T_RUN_RESULTS")
        if send:
            if recipient is None:
                raise click.ClickException("Destinatario non specificato. Usa --recipient <email>.")
            sender_email, sender_password = _get_credentials(ns)
            click.echo(f"[iq report] Eseguo run_ts_portfolio_performance → {recipient} (start={effective_start})")
            out = run_ts_portfolio_performance(
                portfolio_cfg=portfolio_obj,
                sender_email=sender_email,
                sender_password=sender_password,
                recipient_email=recipient,
                show_report=verbose,
                verbose=verbose,
                wfo_results_dir=wfo_dir,
                create_structure=False,
                auto_adjust=True,
                analisys_start_date=effective_start,
                analisys_end_date=end_date,
            )
            click.echo("[iq report] Completato.")
        else:
            click.echo("[iq report] --no-send attivo.")


# ---------------------------------------------------------------------------
# iq analyze
# ---------------------------------------------------------------------------

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio R")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--wfo-results-dir", default=None)
def analyze(ptf, verbose, wfo_results_dir):
    """Lancia analisi R-portfolio (WFO + OFC + MC). Solo R-portfolio."""
    click.echo("[iq analyze] Caricamento librerie...")
    ns = _load_all_libs()

    click.echo(f"[iq analyze] Risolvo portafoglio: {ptf}")
    portfolio_obj, kind = _resolve_portfolio(ptf, ns)

    if kind != "R":
        raise click.ClickException(f"iq analyze supporta solo R-portfolio. '{ptf}' è un K-portfolio.")

    click.echo(f"[iq analyze] {portfolio_obj.get('Title', ptf)} — not implemented yet.")
    click.echo("[iq analyze] Questo comando sarà implementato nella Fase 2 (CLI completa).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
