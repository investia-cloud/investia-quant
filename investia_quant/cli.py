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

_MAIL_ME       = "lf27963@gmail.com"
_MAIL_MANAGERS = ["lf27963@gmail.com", "customercare.ec@gmail.com"]


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


def _load_portfolios_conf() -> list:
    """
    Legge scripts/portfolios.conf e ritorna lista di dict:
      [{"name": str, "type": str, "recipients": list[str]}, ...]
    Cerca in <project_root>/scripts/portfolios.conf, poi in
    IQ_INPUTS_DIR/../scripts/portfolios.conf.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.path.join(root, "scripts", "portfolios.conf")]
    iq_inputs = os.environ.get("IQ_INPUTS_DIR")
    if iq_inputs:
        candidates.append(os.path.normpath(os.path.join(iq_inputs, "..", "scripts", "portfolios.conf")))

    conf_path = next((p for p in candidates if os.path.isfile(p)), None)
    if conf_path is None:
        raise click.ClickException(f"portfolios.conf non trovato in: {candidates}")

    result = []
    with open(conf_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            ptype = parts[1].strip()
            recip_raw = parts[2].strip() if len(parts) > 2 else ""
            recipients = [r.strip() for r in recip_raw.split(",") if r.strip()]
            result.append({"name": name, "type": ptype, "recipients": recipients})
    return result


def _resolve_recipient(mail_args, ptf_recipients: list) -> list:
    """
    Ritorna lista destinatari flat.
    mail_args può essere stringa singola o tupla (multiple=True).
    Shortcut: me → _MAIL_ME, managers → _MAIL_MANAGERS, customers → ptf_recipients da conf.
    """
    if not mail_args:
        return list(ptf_recipients)
    if isinstance(mail_args, str):
        mail_args = (mail_args,)
    result = []
    for arg in mail_args:
        if arg == "customers":
            result.extend(ptf_recipients)
        elif arg == "me":
            result.append(_MAIL_ME)
        elif arg == "managers":
            result.extend(_MAIL_MANAGERS)
        else:
            result.append(arg)
    # deduplicazione mantenendo ordine
    seen = set()
    return [x for x in result if not (x in seen or seen.add(x))]
    
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
@click.option("--ptf", "--portfolio", default=None, help="Nome portafoglio (es. alpha_world, us_trading_2026)")
@click.option("--all", "--ptf-all", "all_portfolios", is_flag=True, default=False, help="Esegui tutti i portafogli da portfolios.conf")
@click.option("--rotational", "--ptf-all-r", is_flag=True, default=False, help="Esegui solo portafogli tipo R (rotational)")
@click.option("--trading", "--ptf-all-k", is_flag=True, default=False, help="Esegui solo portafogli tipo T (trading)")
@click.option("--recipient", "--mail", "--mailto", multiple=True, default=None, help="Destinatario: email, me, managers, customers")
@click.option("--report-date", default=None, help="Data fine report YYYY-MM-DD (default: oggi)")
@click.option("--dry-run", is_flag=True, default=False, help="Simula senza inviare email")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Output verboso")
@click.option("--no-send", is_flag=True, default=False, help="Non inviare email (solo esegui)")
@click.option("--wfo-results-dir", default=None, help="Override directory risultati WFO")
def run(ptf, all_portfolios, rotational, trading, recipient, report_date, dry_run, verbose, no_send, wfo_results_dir):
    """Esecuzione runtime: genera e invia report segnali (R e K portfolio)."""
    select_all = all_portfolios or rotational or trading
    if select_all and ptf:
        raise click.ClickException("Usa --ptf oppure --all/--rotational/--trading, non entrambi.")
    if not select_all and not ptf:
        raise click.ClickException("Specifica --ptf <nome> oppure --all / --rotational / --trading.")

    click.echo("[iq run] Caricamento librerie...")
    ns = _load_all_libs()

    conf_entries = _load_portfolios_conf()
    conf_map = {e["name"]: e for e in conf_entries}

    if ptf:
        ptf_conf = conf_map.get(ptf, {"recipients": []})
        portfolio_obj, kind = _resolve_portfolio(ptf, ns)
        tasks = [(ptf, portfolio_obj, kind, _resolve_recipient(recipient, ptf_conf["recipients"]))]
    else:
        tasks = []
        for entry in conf_entries:
            if rotational or trading:
                if not ((rotational and entry["type"] == "R") or (trading and entry["type"] == "T")):
                    continue
            try:
                portfolio_obj, kind = _resolve_portfolio(entry["name"], ns)
            except click.ClickException as exc:
                click.echo(f"[iq run] SKIP {entry['name']}: {exc}", err=True)
                continue
            tasks.append((entry["name"], portfolio_obj, kind, _resolve_recipient(recipient, entry["recipients"])))

    send_report = not no_send and not dry_run
    sender_email, sender_password = _get_credentials(ns) if send_report else (None, None)

    ok_count = 0
    err_count = 0

    for ptf_name, portfolio_obj, kind, rcpts in tasks:
        rcpts_str = ", ".join(rcpts) if rcpts else "(nessuno)"

        if send_report and not rcpts:
            click.echo(f"[iq run] {ptf_name} ({kind}) → SKIP: nessun destinatario.", err=True)
            err_count += 1
            continue

        try:
            if kind == "R":
                r_run_portfolio = ns.get("r_run_portfolio")
                if r_run_portfolio is None:
                    raise click.ClickException("r_run_portfolio non trovata in r_functions.")
                wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS")
                if send_report:
                    for rcpt in rcpts:
                        if verbose:
                            click.echo(f"[iq run] r_run_portfolio → {rcpt}")
                        out = r_run_portfolio(
                            portfolio=portfolio_obj,
                            report_end_date=report_date,
                            year=None,
                            wfo_results_dir=wfo_dir,
                            sender_email=sender_email,
                            sender_password=sender_password,
                            recipient_email=rcpt,
                            subject=None,
                            verbose=verbose,
                            dry_run=dry_run,
                            debug=verbose,
                        )
                        if verbose:
                            click.echo(f"[iq run] Output: {out}")
                else:
                    if verbose:
                        click.echo(f"[iq run] --no-send attivo: r_run_portfolio NON eseguito ({ptf_name}).")

            elif kind == "K":
                k_run_portfolio = ns.get("k_run_portfolio")
                if k_run_portfolio is None:
                    raise click.ClickException("k_run_portfolio non trovata in k_functions.")
                wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_T_WFO_RESULTS_DIR", "../../inputs/WFO_T_RUN_RESULTS")
                if send_report:
                    for rcpt in rcpts:
                        if verbose:
                            click.echo(f"[iq run] k_run_portfolio → {rcpt}")
                        out = k_run_portfolio(
                            portfolio_cfg=portfolio_obj,
                            report_end_date=report_date,
                            verbose=verbose,
                            wfo_results_dir=wfo_dir,
                            create_structure=False,
                            sender_email=sender_email,
                            sender_password=sender_password,
                            recipient_email=rcpt,
                            subject=None,
                            check_open_trades=False,
                            check_close_trades=False,
                            generate_charts=True,
                            max_attachments_mb=15,
                            max_attachments_count=10,
                            attach_mode="signals_only",
                        )
                        if verbose:
                            click.echo(f"[iq run] Output: {out}")
                else:
                    if verbose:
                        click.echo(f"[iq run] --no-send attivo: k_run_portfolio NON eseguito ({ptf_name}).")

            click.echo(f"[iq run] {ptf_name} ({kind}) → {rcpts_str} ✓")
            ok_count += 1

        except Exception as exc:
            click.echo(f"[iq run] {ptf_name} ({kind}) → ERRORE: {exc}", err=True)
            err_count += 1

    total = ok_count + err_count
    click.echo(f"[iq run] Completato: {ok_count}/{total} portafogli, {err_count} errori.")


# ---------------------------------------------------------------------------
# iq report
# ---------------------------------------------------------------------------

@app.command()
@click.option("--ptf", "--portfolio", default=None, help="Nome portafoglio")
@click.option("--all", "--ptf-all", "all_portfolios", is_flag=True, default=False, help="Esegui tutti i portafogli da portfolios.conf")
@click.option("--rotational", "--ptf-all-r", is_flag=True, default=False, help="Esegui solo portafogli tipo R (rotational)")
@click.option("--trading", "--ptf-all-k", is_flag=True, default=False, help="Esegui solo portafogli tipo T (trading)")
@click.option("--recipient", "--mail", "--mailto", multiple=True, default=None, help="Destinatario: email, me, managers, customers")
@click.option("--start-date", default=None,
              help="Data inizio analisi YYYY-MM-DD (default: 2015-01-01 per R, YTD per K)")
@click.option("--end-date", default=None, help="Data fine analisi YYYY-MM-DD")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--no-send", is_flag=True, default=False, help="Non inviare email")
@click.option("--wfo-results-dir", default=None)
def report(ptf, all_portfolios, rotational, trading, recipient, start_date, end_date, verbose, no_send, wfo_results_dir):
    """Genera e invia report performance (R e K portfolio)."""
    select_all = all_portfolios or rotational or trading
    if select_all and ptf:
        raise click.ClickException("Usa --ptf oppure --all/--rotational/--trading, non entrambi.")
    if not select_all and not ptf:
        raise click.ClickException("Specifica --ptf <nome> oppure --all / --rotational / --trading.")

    click.echo("[iq report] Caricamento librerie...")
    ns = _load_all_libs()

    conf_entries = _load_portfolios_conf()
    conf_map = {e["name"]: e for e in conf_entries}

    if ptf:
        ptf_conf = conf_map.get(ptf, {"recipients": []})
        portfolio_obj, kind = _resolve_portfolio(ptf, ns)
        tasks = [(ptf, portfolio_obj, kind, _resolve_recipient(recipient, ptf_conf["recipients"]))]
    else:
        tasks = []
        for entry in conf_entries:
            if rotational or trading:
                if not ((rotational and entry["type"] == "R") or (trading and entry["type"] == "T")):
                    continue
            try:
                portfolio_obj, kind = _resolve_portfolio(entry["name"], ns)
            except click.ClickException as exc:
                click.echo(f"[iq report] SKIP {entry['name']}: {exc}", err=True)
                continue
            tasks.append((entry["name"], portfolio_obj, kind, _resolve_recipient(recipient, entry["recipients"])))

    send = not no_send
    sender_email, sender_password = _get_credentials(ns) if send else (None, None)

    for ptf_name, portfolio_obj, kind, rcpts in tasks:
        rcpts_str = ", ".join(rcpts) if rcpts else "(nessuno)"
        click.echo(f"[iq report] Portafoglio: {ptf_name} ({kind}) — destinatari: {rcpts_str}")

        if send and not rcpts:
            click.echo(f"[iq report] SKIP {ptf_name}: nessun destinatario.", err=True)
            continue

        if kind == "R":
            effective_start = start_date if start_date is not None else _DEFAULT_ANALYSIS_START_R
            run_rotational_portfolio_performance = ns.get("run_rotational_portfolio_performance")
            if run_rotational_portfolio_performance is None:
                raise click.ClickException("run_rotational_portfolio_performance non trovata in r_functions.")
            wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS")
            if send:
                for rcpt in rcpts:
                    click.echo(f"[iq report] run_rotational_portfolio_performance → {rcpt} (start={effective_start})")
                    out = run_rotational_portfolio_performance(
                        portfolio=portfolio_obj,
                        analisys_start_date=effective_start,
                        analisys_end_date=end_date,
                        wfo_results_dir=wfo_dir,
                        sender_email=sender_email,
                        sender_password=sender_password,
                        recipient_email=rcpt,
                        show_report=verbose,
                        debug=verbose,
                        verbose=verbose,
                        auto_adjust=True,
                    )
                    click.echo(f"[iq report] Completato ({ptf_name} → {rcpt}).")
            else:
                click.echo(f"[iq report] --no-send attivo ({ptf_name}).")

        elif kind == "K":
            effective_start = start_date if start_date is not None else _DEFAULT_ANALYSIS_START_K
            run_ts_portfolio_performance = ns.get("run_ts_portfolio_performance")
            if run_ts_portfolio_performance is None:
                raise click.ClickException("run_ts_portfolio_performance non trovata in k_functions.")
            wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_T_WFO_RESULTS_DIR", "../../inputs/WFO_T_RUN_RESULTS")
            if send:
                for rcpt in rcpts:
                    click.echo(f"[iq report] run_ts_portfolio_performance → {rcpt} (start={effective_start})")
                    out = run_ts_portfolio_performance(
                        portfolio_cfg=portfolio_obj,
                        sender_email=sender_email,
                        sender_password=sender_password,
                        recipient_email=rcpt,
                        show_report=verbose,
                        verbose=verbose,
                        wfo_results_dir=wfo_dir,
                        create_structure=False,
                        auto_adjust=True,
                        analisys_start_date=None,
                        analisys_end_date=end_date,
                    )
                    click.echo(f"[iq report] Completato ({ptf_name} → {rcpt}).")
            else:
                click.echo(f"[iq report] --no-send attivo ({ptf_name}).")


# ---------------------------------------------------------------------------
# iq analyze
# ---------------------------------------------------------------------------

@app.command()
@click.option("--ptf", default=None,
              help="Nome portafoglio R da registry (es. alpha_fact)")
@click.option("--universe", default=None,
              help="CSV con colonna 'ticker' — universo ad hoc")
@click.option("--output-dir", default=None,
              help="Directory output PDF + PNG (default: outputs/reports/<nome>/<data>/)")
@click.option("--profile", default="satellite",
              type=click.Choice(["satellite", "core"]),
              help="Profilo OFC: satellite (default) o core")
@click.option("--year", default=None, type=int,
              help="Anno selezione WFO (default: anno corrente)")
@click.option("--start-date", default="2015-01-01",
              help="Inizio storico download (default: 2015-01-01)")
@click.option("--verbose", is_flag=True, default=False)
def analyze(ptf, universe, output_dir, profile, year, start_date, verbose):
    """Lancia analisi R-portfolio (WFO + OFC + MC). Solo R-portfolio."""

    # Validazione input
    if ptf and universe:
        raise click.ClickException("Usa --ptf oppure --universe, non entrambi.")
    if not ptf and not universe:
        raise click.ClickException("Specifica --ptf <nome> oppure --universe <file.csv>.")

    click.echo("[iq analyze] Caricamento librerie...")
    ns = _load_all_libs()

    run_r_portfolio_analysis = ns.get("run_r_portfolio_analysis")
    if run_r_portfolio_analysis is None:
        raise click.ClickException("run_r_portfolio_analysis non trovata in r_functions.")

    # Risolvi portfolio_cfg
    if ptf:
        portfolio_obj, kind = _resolve_portfolio(ptf, ns)
        if kind != "R":
            raise click.ClickException(
                f"iq analyze supporta solo R-portfolio. '{ptf}' è un K-portfolio."
            )
        ptf_name = ptf
    else:
        # --universe: carica CSV e costruisce cfg sintetico
        import csv
        from pathlib import Path as _Path
        universe_path = _Path(universe)
        if not universe_path.exists():
            raise click.ClickException(f"File universe non trovato: {universe}")
        with open(universe_path) as f:
            reader = csv.DictReader(f)
            if "ticker" not in (reader.fieldnames or []):
                raise click.ClickException("Il CSV deve avere colonna 'ticker'.")
            tickers_list = [row["ticker"].strip() for row in reader if row["ticker"].strip()]
        if not tickers_list:
            raise click.ClickException("Nessun ticker trovato nel CSV.")
        ptf_name = universe_path.stem
        portfolio_obj = {
            "Title":              ptf_name,
            "tickers":            tickers_list,
            "benchmark_portfolio": None,
            "benchmark_title":    None,
            "risk_off_tickers":   [],
        }

    # Risolvi output_dir
    if output_dir is None:
        import datetime
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        today = datetime.date.today().isoformat()
        output_dir = os.path.join(root, "outputs", "reports", ptf_name, today)

    click.echo(f"[iq analyze] Portafoglio: {portfolio_obj.get('Title', ptf_name)}")
    click.echo(f"[iq analyze] Output dir:  {output_dir}")
    click.echo(f"[iq analyze] Profile:     {profile}")
    click.echo(f"[iq analyze] Avvio pipeline (WFO + OFC + MC)...")

    try:
        result = run_r_portfolio_analysis(
            portfolio_cfg = portfolio_obj,
            output_dir    = output_dir,
            year          = year,
            start_date    = start_date,
            end_date      = None,
            profile       = profile,
            verbose       = verbose,
        )
        click.echo(f"[iq analyze] Completato.")
        click.echo(f"  PDF:           {result['pdf']}")
        click.echo(f"  Plots dir:     {result['plots_dir']}")
        click.echo(f"  OFC Standard:  {'PROMOTED' if result['ofc_std'] else 'REJECTED'}")
        click.echo(f"  OFC Cluster:   {'PROMOTED' if result['ofc_cluster'] else 'REJECTED'}")
        click.echo(f"  Skill Std:     {result['skill_profile_std']}")
        click.echo(f"  Skill Cluster: {result['skill_profile_cluster']}")
    except Exception as exc:
        raise click.ClickException(f"Pipeline fallita: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
