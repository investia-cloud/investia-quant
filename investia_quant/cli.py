import click

@click.group()
def app():
    """investia-quant CLI"""
    pass

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio")
def run(ptf):
    """Esecuzione runtime (R e K portfolio)"""
    click.echo(f"[iq run] {ptf} — not implemented")

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio")
def report(ptf):
    """Genera PDF relazione tecnica"""
    click.echo(f"[iq report] {ptf} — not implemented")

@app.command()
@click.option("--ptf", required=True, help="Nome portafoglio")
def analyze(ptf):
    """Lancia analisi R-portfolio"""
    click.echo(f"[iq analyze] {ptf} — not implemented")
