"""
investia_quant/cli.py — CLI `iq`
Entry point: iq run / iq report / iq r-analyze / iq l-analyze / iq k-analyze / iq k-agent
"""

import sys
import os
import datetime
import shutil
from pathlib import Path
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
    Risolve --ptf <nome> cercando in R_PORTFOLIO_REGISTRY, K_PORTFOLIO_REGISTRY
    e L_PORTFOLIO_REGISTRY.
    Ritorna (portfolio_obj, kind) dove kind è 'R', 'K' o 'L'.
    """
    r_registry = ns.get("R_PORTFOLIO_REGISTRY", {})
    k_registry = ns.get("K_PORTFOLIO_REGISTRY", {})
    l_registry = ns.get("L_PORTFOLIO_REGISTRY", {})

    if ptf_name in r_registry:
        return r_registry[ptf_name], "R"
    if ptf_name in k_registry:
        return k_registry[ptf_name], "K"
    if ptf_name in l_registry:
        return l_registry[ptf_name], "L"

    # Fallback: cerca direttamente per nome variabile nel namespace
    obj = ns.get(ptf_name)
    if obj is not None and isinstance(obj, dict):
        if "trading_systems" in obj:
            return obj, "K"
        if "tickers" in obj:
            # R: tickers è una lista di ticker (motore rotazionale, no pesi fissi)
            # L nuovo formato: tickers è un dict {ticker: peso} con somma ~1.0
            if isinstance(obj["tickers"], list):
                return obj, "R"
            return obj, "L"
        # dict semplice {ticker: peso} -> Lazy portfolio vecchio formato
        return obj, "L"

    available_r = list(r_registry.keys())
    available_k = list(k_registry.keys())
    available_l = list(l_registry.keys())
    raise click.ClickException(
        f"Portafoglio '{ptf_name}' non trovato.\n"
        f"  R-portfolio disponibili: {available_r}\n"
        f"  K-portfolio disponibili: {available_k}\n"
        f"  L-portfolio disponibili: {available_l}"
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
        return []
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


def _expand_ptf_names(ptf_str: str, registry: dict):
    """
    Espande una stringa --ptf in una lista di nomi portafoglio.
    Supporta: singolo nome, lista separata da spazi, pattern glob (*, ?, []).
    Ritorna (names: list[str], errors: list[str]).
    """
    import fnmatch
    tokens = ptf_str.split()
    seen = {}
    errors = []
    for token in tokens:
        if any(c in token for c in '*?['):
            matches = [name for name in registry if fnmatch.fnmatch(name, token)]
            if not matches:
                errors.append(f"Pattern '{token}' non ha trovato nessun Lazy portfolio corrispondente.")
            for m in matches:
                seen[m] = None
        else:
            seen[token] = None
    return list(seen.keys()), errors


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

@app.command(epilog=(
    "\b\nEsempi:\n"
    "  iq run --ptf alpha_world --mail me\n"
    "  iq run --rotational --mail managers --mail customers\n"
    "  iq run --all --verbose\n"
))
@click.option("--ptf", "--portfolio", default=None, help="Nome portafoglio (es. alpha_world, us_trading_2026)")
@click.option("--all", "--ptf-all", "all_portfolios", is_flag=True, default=False, help="Esegui tutti i portafogli da portfolios.conf")
@click.option("--rotational", "--ptf-all-r", is_flag=True, default=False, help="Esegui solo portafogli tipo R (rotational)")
@click.option("--trading", "--ptf-all-k", is_flag=True, default=False, help="Esegui solo portafogli tipo T (trading)")
@click.option("--recipient", "--mail", "--mailto", multiple=True, default=None, help="Destinatario: email, me, managers, customers (facoltativo — se omesso la pipeline gira ma nessuna mail viene inviata)")
@click.option("--report-date", default=None, help="Data fine report YYYY-MM-DD (default: oggi)")
@click.option("--dry-run", is_flag=True, default=False, help="Simula senza inviare email")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Output verboso")
@click.option("--wfo-results-dir", default=None, help="Override directory risultati WFO")
def run(ptf, all_portfolios, rotational, trading, recipient, report_date, dry_run, verbose, wfo_results_dir):
    """Esecuzione runtime: genera e invia report segnali (R e K portfolio).

    Seleziona un singolo PTF (--ptf) oppure un gruppo (--all/--rotational/--trading)
    da portfolios.conf. Se --mail è omesso la pipeline viene eseguita normalmente
    ma nessuna email viene inviata.
    """
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

    send_email = bool(recipient)
    sender_email, sender_password = _get_credentials(ns) if send_email else (None, None)

    ok_count = 0
    err_count = 0

    for ptf_name, portfolio_obj, kind, rcpts in tasks:
        rcpts_str = ", ".join(rcpts) if rcpts else "(nessuno — pipeline senza invio)"

        try:
            if kind == "R":
                r_run_portfolio = ns.get("r_run_portfolio")
                if r_run_portfolio is None:
                    raise click.ClickException("r_run_portfolio non trovata in r_functions.")
                wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS")
                if rcpts:
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
                        click.echo(f"[iq run] --mail non specificato: pipeline eseguita senza invio email ({ptf_name}).")
                    out = r_run_portfolio(
                        portfolio=portfolio_obj,
                        report_end_date=report_date,
                        year=None,
                        wfo_results_dir=wfo_dir,
                        sender_email="",
                        sender_password="",
                        recipient_email="",
                        subject=None,
                        verbose=verbose,
                        dry_run=dry_run,
                        debug=verbose,
                    )
                    if verbose:
                        click.echo(f"[iq run] Output: {out}")

            elif kind == "K":
                k_run_portfolio = ns.get("k_run_portfolio")
                if k_run_portfolio is None:
                    raise click.ClickException("k_run_portfolio non trovata in k_functions.")
                wfo_dir = wfo_results_dir or ns.get("_TSLAB_RUNTIME_T_WFO_RESULTS_DIR", "../../inputs/WFO_T_RUN_RESULTS")
                if rcpts:
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
                        click.echo(f"[iq run] --mail non specificato: pipeline eseguita senza invio email ({ptf_name}).")
                    out = k_run_portfolio(
                        portfolio_cfg=portfolio_obj,
                        report_end_date=report_date,
                        verbose=verbose,
                        wfo_results_dir=wfo_dir,
                        create_structure=False,
                        sender_email="",
                        sender_password="",
                        recipient_email="",
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

@app.command(epilog=(
    "\b\nEsempi:\n"
    "  iq report --ptf alpha_world --mail me\n"
    "  iq report --rotational --start-date 2018-01-01 --mail managers\n"
    "  iq report --all --no-send\n"
))
@click.option("--ptf", "--portfolio", default=None, help="Nome portafoglio")
@click.option("--all", "--ptf-all", "all_portfolios", is_flag=True, default=False, help="Esegui tutti i portafogli da portfolios.conf")
@click.option("--rotational", "--ptf-all-r", is_flag=True, default=False, help="Esegui solo portafogli tipo R (rotational)")
@click.option("--trading", "--ptf-all-k", is_flag=True, default=False, help="Esegui solo portafogli tipo T (trading)")
@click.option("--recipient", "--mail", "--mailto", multiple=True, default=None, help="Destinatario: email, me, managers, customers")
@click.option("--start-date", default=None,
              help="Data inizio analisi YYYY-MM-DD (default: 2015-01-01 per R, YTD per K)")
@click.option("--end-date", default=None, help="Data fine analisi YYYY-MM-DD")
@click.option("--verbose", is_flag=True, default=False, help="Output verboso")
@click.option("--no-send", is_flag=True, default=False, help="Non inviare email")
@click.option("--wfo-results-dir", default=None, help="Override directory risultati WFO")
def report(ptf, all_portfolios, rotational, trading, recipient, start_date, end_date, verbose, no_send, wfo_results_dir):
    """Genera e invia report performance storica (R e K portfolio).

    Seleziona un singolo PTF (--ptf) o un gruppo (--all/--rotational/--trading)
    e calcola le metriche di performance sul periodo indicato.
    """
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
            _tickers = portfolio_obj.get("tickers")
            if isinstance(_tickers, str):
                from datetime import datetime as _dt
                _year = _dt.now().year
                _by_year_map = {
                    "sp100": "alpha_sp100_tickers_by_year",
                    "nasdaq100": "alpha_nasdaq100_tickers_by_year",
                }
                _map_name = _by_year_map.get(_tickers)
                if _map_name is None:
                    raise click.ClickException(
                        f"Portfolio '{ptf_name}': tickers='{_tickers}' non riconosciuto "
                        f"(attesi: {list(_by_year_map.keys())})."
                    )
                _by_year = ns.get(_map_name, {})
                _resolved = _by_year.get(_year) or (_by_year.get(max(_by_year.keys())) if _by_year else None)
                if not _resolved:
                    raise click.ClickException(
                        f"Portfolio '{ptf_name}': nessun mapping ticker trovato per "
                        f"anno {_year} in {_map_name}."
                    )
                portfolio_obj = dict(portfolio_obj)
                portfolio_obj["tickers"] = _resolved
                if verbose:
                    click.echo(f"[iq report] '{ptf_name}': tickers risolti da '{_tickers}' "
                               f"({_map_name}[{_year}]) -> {len(_resolved)} ticker.")
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
# iq r-analyze
# ---------------------------------------------------------------------------

@app.command("r-analyze", epilog=(
    "\b\nEsempi:\n"
    "  iq r-analyze --ptf alpha_fact\n"
    "  iq r-analyze --ptf alpha_fact --relazione-tecnica\n"
    "  iq r-analyze --ptf alpha_fact --engine Momentum\n"
    "  iq r-analyze --universe inputs/universe.csv --profile core --relazione-tecnica\n"
))
@click.option("--ptf", default=None,
              help="Nome portafoglio R da registry (es. alpha_fact)")
@click.option("--universe", default=None,
              help="CSV con colonna 'ticker' — universo ad hoc")
@click.option("--output-dir", default=None,
              help="Directory output PDF + PNG (default: outputs/r_analysis/<nome>/<timestamp>/)")
@click.option("--profile", default="satellite",
              type=click.Choice(["satellite", "core"]),
              help="Profilo OFC: satellite (default) o core")
@click.option("--year", default=None, type=int,
              help="Anno selezione WFO (default: anno corrente)")
@click.option("--start-date", default="2015-01-01",
              help="Inizio storico download (default: 2015-01-01)")
@click.option("--relazione-tecnica", "relazione_tecnica", is_flag=True, default=False,
              help=(
                  "Genera la relazione tecnica completa (analisi LLM + PTF card .md + PDF). "
                  "Senza questo flag: solo pipeline WFO+OFC+MC, nessuna chiamata LLM."
              ))
@click.option("--engine", default=None,
              type=click.Choice(["Momentum", "Multifactor"]),
              help="Engine WFO da eseguire (default: entrambi Momentum e Multifactor)")
@click.option("--verbose", is_flag=True, default=False, help="Output verboso")
def r_analyze(ptf, universe, output_dir, profile, year, start_date, relazione_tecnica, engine, verbose):
    """Pipeline R-portfolio N-engine: WFO + OFC + MC. Solo R-portfolio.

    Esegue sempre la pipeline WFO+OFC+MC. Con --relazione-tecnica genera
    anche l'analisi LLM, la PTF card .md e la relazione tecnica PDF
    (sempre insieme — blocco atomico).
    """

    # Validazione input
    if ptf and universe:
        raise click.ClickException("Usa --ptf oppure --universe, non entrambi.")
    if not ptf and not universe:
        raise click.ClickException("Specifica --ptf <nome> oppure --universe <file.csv>.")

    click.echo("[iq r-analyze] Caricamento librerie...")
    ns = _load_all_libs()

    run_analysis = ns.get("run_r_portfolio_n_engine_analysis")
    if run_analysis is None:
        raise click.ClickException("run_r_portfolio_n_engine_analysis non trovata in r_functions.")

    # Risolvi portfolio_cfg
    if ptf:
        portfolio_obj, kind = _resolve_portfolio(ptf, ns)
        if kind != "R":
            raise click.ClickException(
                f"iq r-analyze supporta solo R-portfolio. '{ptf}' è un K-portfolio."
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
        _get_dir = ns.get("get_analysis_output_dir")
        output_dir = str(_get_dir("r_analysis", ptf_name=ptf_name, profilo=profile))

    engines_to_run = [engine] if engine else ["Momentum", "Multifactor"]

    click.echo(f"[iq r-analyze] Portafoglio:         {portfolio_obj.get('Title', ptf_name)}")
    click.echo(f"[iq r-analyze] Output dir:          {output_dir}")
    click.echo(f"[iq r-analyze] Profile:             {profile}")
    click.echo(f"[iq r-analyze] Engine(s):           {', '.join(engines_to_run)}")
    click.echo(f"[iq r-analyze] Relazione tecnica:   {'sì' if relazione_tecnica else 'no (usa --relazione-tecnica)'}")
    click.echo(f"[iq r-analyze] Avvio pipeline (WFO + OFC + MC)...")

    try:
        result = run_analysis(
            portfolio_cfg      = portfolio_obj,
            output_dir         = output_dir,
            year               = year,
            start_date         = start_date,
            end_date           = None,
            profile            = profile,
            verbose            = verbose,
            relazione_tecnica  = relazione_tecnica,
            engines            = engines_to_run,
        )
        click.echo(f"[iq r-analyze] Completato.")
        click.echo(f"  Plots dir:     {result['plots_dir']}")
        for eng_name, eng_data in result.get("engines", {}).items():
            ofc_promoted = bool((eng_data.get("ofc_report") or {}).get("promoted", False))
            skill = eng_data.get("skill_profile", "N/A")
            verdict = "PROMOTED" if ofc_promoted else "REJECTED"
            click.echo(f"  OFC {eng_name:<12}: {verdict}  (skill: {skill})")
        if relazione_tecnica:
            click.echo(f"  Card MD:       {result['card_path'] or '(non generato)'}")
            click.echo(f"  PDF:           {result['pdf_path'] or '(non generato)'}")
    except Exception as exc:
        raise click.ClickException(f"Pipeline fallita: {exc}")


# ---------------------------------------------------------------------------
# iq k-analyze
# ---------------------------------------------------------------------------

@app.command("k-analyze", epilog=(
    "\b\nEsempi:\n"
    "  iq k-analyze -s dbma_matrix -t NVDA\n"
    "  iq k-analyze --ptf us_trading_2026\n"
    "  iq k-analyze -s dbma_matrix bollinger -t NVDA AAPL --override\n"
))
@click.option("-s", "--strategies", multiple=True, default=None,
    help="Una o più strategie (es. -s dbma_matrix bollinger)")
@click.option("-t", "--tickers", multiple=True, default=None,
    help="Uno o più ticker (es. -t NVDA AAPL)")
@click.option("--ptf", default=None,
    help="Nome K-portfolio (es. us_trading_2026) — estrae tickers automaticamente")
@click.option("--output-dir", default=None,
    help="Directory output (default: outputs/k_analysis/<timestamp>/)")
@click.option("--start-date", default="2015-01-01", show_default=True,
    help="Inizio storico download")
@click.option("--end-date", default=None, help="Fine storico (default: oggi)")
@click.option("--ratio", default="4:1", show_default=True,
    help="Train:test ratio WFO")
@click.option("--fees", default=0.001, show_default=True,
    help="Commissioni per trade")
@click.option("--slippage", default=0.002, show_default=True,
    help="Slippage per trade")
@click.option("--price-col", default="Open", show_default=True,
    help="Colonna prezzo OHLCV")
@click.option("--selection-metric", default="total_return", show_default=True,
    help="Metrica selezione parametri WFO")
@click.option("--init-cash", default=100_000.0, show_default=True,
    help="Capitale iniziale")
@click.option("--warmup-years", default=1, show_default=True,
    help="Anni warmup WFO")
@click.option("--wfo-results-dir", default=None,
    help="Directory risultati WFO (default: outputs/WFO_T_DEV_RESULTS/)")
@click.option("--override", is_flag=True, default=False,
    help="Ricalcola risultati WFO già salvati")
@click.option("--n-simulations", default=1_000, show_default=True,
    help="Numero simulazioni Monte Carlo")
@click.option("--block-size", default=10, show_default=True,
    help="Block size per Block Bootstrap MC")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Output verboso")
def k_analyze(strategies, tickers, ptf, output_dir, start_date, end_date,
              ratio, fees, slippage, price_col, selection_metric, init_cash,
              warmup_years, wfo_results_dir, override, n_simulations, block_size,
              verbose):
    """Analisi K-strategy: inspector (1×1) o panel (N×M) in base agli argomenti."""
    def _parse_list(vals):
        if not vals:
            return None
        result = []
        for v in vals:
            result.extend(v.split())
        return result or None

    if ptf and tickers:
        raise click.ClickException("--ptf e -t sono mutuamente esclusivi.")
    if not ptf and not tickers:
        raise click.ClickException("Specifica --ptf <nome> oppure -t <tickers>.")

    ns = _load_all_libs()

    if ptf:
        ptf_obj, kind = _resolve_portfolio(ptf, ns)
        if kind != "K":
            raise click.ClickException(f"'{ptf}' è un R-portfolio. iq k-analyze accetta solo K-portfolio.")
        t = list({ts["symbol"] for ts in ptf_obj.get("trading_systems", [])})
        if not t:
            raise click.ClickException(f"Nessun ticker trovato in '{ptf}'.")
        if verbose:
            print(f"PTF '{ptf}': {len(t)} tickers → {t}")
    else:
        t = _parse_list(tickers)

    # Strategie: esplicite se -s passato, altrimenti tutte disponibili
    s = _parse_list(strategies) if strategies else None
    if s is None:
        # Tutte le strategie disponibili nel namespace
        s = [
            k.replace("strategy_", "").replace("_param_ranges", "")
            for k in ns
            if k.startswith("strategy_") and k.endswith("_param_ranges")
        ]
        if verbose:
            print(f"Strategie disponibili: {len(s)}")

    if len(s) == 1 and len(t) == 1:
        print(f"Inspector: {s[0]}@{t[0]}")
    else:
        print(f"Panel: {len(s)} strategie × {len(t)} ticker")

    _get_dir = ns.get("get_analysis_output_dir")
    out_dir = output_dir or str(_get_dir("k_analysis"))
    import importlib.util, sys as _sys
    lib = Path(__file__).parent.parent / "notebooks" / "libs_py" / "k_functions.py"
    spec = importlib.util.spec_from_file_location("k_functions", lib)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.run_k_strategy_analysis(
        strategies=s,
        tickers=t,
        output_dir=out_dir,
        start_date=start_date,
        end_date=end_date,
        scenario="B",
        verbose=verbose,
        ratio=ratio,
        fees=fees,
        slippage=slippage,
        price_col=price_col,
        selection_metric=selection_metric,
        init_cash=init_cash,
        warmup_years=warmup_years,
        save_results=True,
        wfo_results_dir=wfo_results_dir,
        override=override,
        n_simulations=n_simulations,
        block_size=block_size,
    )
    print(f"Modalità  : {result['mode']}")
    print(f"Promossi  : {len(result['promoted'])}")
    print(f"Output    : {result['plots_dir']}")
    if result['promoted']:
        print("Coppie promosse:")
        for ticker, strategy in result['promoted']:
            print(f"  {ticker} @ {strategy}")


# ---------------------------------------------------------------------------
# iq l-analyze
# ---------------------------------------------------------------------------

@app.command("l-analyze", epilog=(
    "\b\nEsempi:\n"
    "  iq l-analyze --ptf lazy_etf_port\n"
    "  iq l-analyze --ptf lazy_etf_port --pdf\n"
    "  iq l-analyze --ptf all --override\n"
    "\nCon --pdf: genera la Relazione Investitore solo per PTF PROMOSSI;\n"
    "  i RIGETTATI stampano un messaggio e non producono PDF.\n"
    "  Output: outputs/l_analysis/<timestamp>/<ptf>/<ptf>_Relazione_Investitore.pdf\n"
))
@click.option("--ptf", default=None,
    help="Nome Lazy portfolio, oppure 'all' per tutti i PTF nel registry")
@click.option("--output-dir", default=None,
    help="Directory output (default: outputs/l_analysis/<timestamp>/)")
@click.option("--start-date", default="2016-01-01", show_default=True,
    help="Inizio storico backtest")
@click.option("--end-date", default=None, help="Fine storico (default: oggi)")
@click.option("--benchmark", default="SPY", show_default=True,
    help="Ticker benchmark per il confronto")
@click.option("--init-cash", default=100_000.0, show_default=True,
    help="Capitale iniziale backtest")
@click.option("--fees", default=0.001, show_default=True, help="Commissioni per trade")
@click.option("--years", default=10, show_default=True,
    help="Anni per frontiera efficiente e stability test")
@click.option("--n-simulations-mc-a", default=1000, show_default=True,
    help="Simulazioni Monte Carlo Block A")
@click.option("--n-simulations-mc-b", default=500, show_default=True,
    help="Simulazioni Monte Carlo Block B")
@click.option("--override", is_flag=True, default=False,
    help="Ricalcola anche i PTF già in cache (default: skip se cache presente)")
@click.option("--min-years", default=5, show_default=True,
    help="Storico minimo comune richiesto (anni) per run_bh_backtest e MC Block B")
@click.option("--pdf", "gen_pdf", is_flag=True, default=False,
    help="Genera la Relazione Investitore PDF per i PTF promossi (default: solo pipeline)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Output verboso")
def l_analyze(ptf, output_dir, start_date, end_date, benchmark,
              init_cash, fees, years, n_simulations_mc_a,
              n_simulations_mc_b, override, min_years, gen_pdf, verbose):
    """Pipeline Lazy portfolio: frontiera + backtest + stability + MC A/B + DSR.

    Batch (--ptf all) o singolo. Con --pdf genera la Relazione Investitore per
    ogni PTF PROMOSSO in outputs/l_analysis/<timestamp>/<ptf>/<ptf_name>_Relazione_Investitore.pdf.
    I PTF RIGETTATI producono solo un messaggio — nessun PDF investitore viene generato.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Silenzia il logger di yfinance (failed download via logger.error)
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    if not ptf:
        raise click.ClickException("Specifica --ptf <nome> oppure --ptf all.")

    ns = _load_all_libs()
    l_registry = ns.get("L_PORTFOLIO_REGISTRY", {})

    if ptf.lower() == "all":
        l_lazy_registry = ns.get("L_PORTFOLIO_LAZY", {})
        if not l_lazy_registry:
            # Fallback: se L_PORTFOLIO_LAZY non esiste (versione vecchia
            # di l_portfolios.py), usa il registry generale con warning
            click.echo("[WARN] L_PORTFOLIO_LAZY non trovato, uso L_PORTFOLIO_REGISTRY completo.", err=True)
            l_lazy_registry = l_registry
        ptf_names = list(l_lazy_registry.keys())
        if not ptf_names:
            raise click.ClickException("L_PORTFOLIO_LAZY è vuoto.")
        if verbose:
            click.echo(f"[iq l-analyze] --ptf all userà solo categoria 'lazy_': {len(ptf_names)} PTF.")
    else:
        _lazy_reg_for_glob = ns.get("L_PORTFOLIO_LAZY") or l_registry
        ptf_names, _glob_errors = _expand_ptf_names(ptf, _lazy_reg_for_glob)
        if _glob_errors:
            raise click.ClickException("\n".join(_glob_errors))
        for _name in ptf_names:
            _obj, _kind = _resolve_portfolio(_name, ns)
            if _kind != "L":
                raise click.ClickException(f"'{_name}' non è un Lazy portfolio (kind={_kind}).")

    if verbose:
        print(f"Lazy-analyze: {len(ptf_names)} PTF -> {ptf_names}")

    _get_dir = ns.get("get_analysis_output_dir")
    out_dir = output_dir or str(_get_dir("l_analysis"))

    run_lazy_portfolio_analysis_fn = ns.get("run_lazy_portfolio_analysis")
    if run_lazy_portfolio_analysis_fn is None:
        raise click.ClickException(
            "run_lazy_portfolio_analysis non trovata in mc_functions.py "
            "(verifica che sia stata aggiunta e che _load_all_libs() "
            "abbia importato mc_functions correttamente)."
        )

    result = run_lazy_portfolio_analysis_fn(
        registry=l_registry,
        ptf_names=ptf_names,
        output_dir=out_dir,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        init_cash=init_cash,
        fees=fees,
        years=years,
        n_simulations_mc_a=n_simulations_mc_a,
        n_simulations_mc_b=n_simulations_mc_b,
        override=override,
        verbose=verbose,
        min_years=min_years,
        generate_pdf=gen_pdf,
    )

    df = result['df']
    print(f"\n[iq l-analyze] Completato. {len(df)} PTF analizzati.")
    print(f"Output dir: {out_dir}")
    print(df.to_string(index=False))
    n_promoted = (df["Verdetto"] == "PROMOSSO").sum() if "Verdetto" in df.columns else 0
    print(f"\nPromossi: {n_promoted}/{len(df)}")


# ---------------------------------------------------------------------------
# iq k-agent
# ---------------------------------------------------------------------------

@app.command("k-agent", epilog=(
    "\b\nEsempi:\n"
    "  iq k-agent --max 3\n"
    "  iq k-agent --pdf paper.pdf --llm anthropic\n"
    "  iq k-agent --llm ollama --model qwen2.5-coder:7b -v\n"
))
@click.option("--max", "max_per_run", default=None, type=int,
    help="Numero massimo di articoli da processare per run (default: 5, mutuamente esclusivo con --pdf)")
@click.option("--pdf", "pdf_path", type=click.Path(exists=True), default=None,
    help="Processa un PDF locale invece del feed RSS")
@click.option("--llm", "llm_provider", default="anthropic",
    type=click.Choice(["ollama", "anthropic"]),
    show_default=True, help="Provider LLM")
@click.option("--model", default=None,
    help="Modello LLM (default: qwen2.5-coder:7b per ollama, claude-sonnet-4-20250514 per anthropic)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Output verboso")
def k_agent(max_per_run, pdf_path, llm_provider, model, verbose):
    """Genera nuove K-strategy leggendo articoli da feed RSS."""
    if pdf_path is not None and max_per_run is not None:
        click.echo("Errore: --pdf e --max sono mutuamente esclusivi.", err=True)
        raise SystemExit(1)
    if max_per_run is None:
        max_per_run = 5

    import sys as _sys
    agent_dir = str(Path(__file__).parent.parent / "K-Strategy-Agent")
    if agent_dir not in _sys.path:
        _sys.path.insert(0, agent_dir)
    import agent as _agent

    _agent.MAX_PER_RUN = max_per_run
    _agent.LLM_PROVIDER = llm_provider
    if model:
        if llm_provider == "ollama":
            _agent.OLLAMA_MODEL = model
        else:
            _agent.ANTHROPIC_MODEL = model
    if verbose:
        print(f"Provider : {llm_provider}")
        print(f"Max      : {max_per_run}")
        print(f"Model    : {model or '(default)'}")
        if pdf_path:
            print(f"PDF      : {pdf_path}")

    if pdf_path:
        _agent.run_agent_from_pdf(pdf_path)
    else:
        _agent.run_agent()


# ---------------------------------------------------------------------------
# iq promote
# ---------------------------------------------------------------------------

@app.command("promote", epilog=(
    "\b\nEsempi:\n"
    "  iq promote --ptf alpha_nasdaq100 --year 2026\n"
    "  iq promote --ptf alpha_nasdaq100 --year 2026 --force\n"
))
@click.option("--ptf", required=True, help="Nome portafoglio R (slug, es. alpha_nasdaq100)")
@click.option("--year", default=None, type=int, help="Anno (default: anno corrente)")
@click.option("--force", is_flag=True, default=False, help="Sovrascrive il file runtime se già esistente")
def promote(ptf, year, force):
    """Copia il file WFO deciso (JN §8) da dev a runtime per il deploy."""
    ns = _load_all_libs()

    portfolio_obj, kind = _resolve_portfolio(ptf, ns)
    if kind != "R":
        raise click.ClickException(
            f"iq promote supporta solo R-portfolio. '{ptf}' è un {kind}-portfolio."
        )

    portfolio_title = portfolio_obj.get("Title", ptf)
    if year is None:
        year = datetime.datetime.now().year

    dev_dir  = Path(ns.get("_TSLAB_DEV_R_WFO_RESULTS_DIR",  "../../outputs/WFO_R_DEV_RESULTS"))
    run_dir  = Path(ns.get("_TSLAB_RUNTIME_R_WFO_RESULTS_DIR", "../../inputs/WFO_R_RUN_RESULTS"))
    filename = f"{portfolio_title}_{year}.wfo_summary.csv"
    src      = dev_dir / filename
    dst      = run_dir / filename

    if not src.exists():
        raise click.ClickException(
            f"File sorgente non trovato:\n  {src}\n"
            "Esegui prima il JN §8 per produrre il file WFO."
        )

    # Riepilogo pre-copia
    src_stat = src.stat()
    click.echo(f"[iq promote] Portafoglio : {portfolio_title} ({year})")
    click.echo(f"[iq promote] Origine     : {src}")
    click.echo(f"[iq promote] Destinazione: {dst}")
    click.echo(f"[iq promote] Dimensione  : {src_stat.st_size} byte  |  Modificato: "
               f"{datetime.datetime.fromtimestamp(src_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    if dst.exists() and not force:
        raise click.ClickException(
            f"Il file runtime esiste già:\n  {dst}\n"
            "Usa --force per sovrascriverlo (es. dopo una revisione intra-anno)."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    click.echo(f"[iq promote] Copiato: {src} -> {dst}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
