# Mantenere equivalenti Notebook e CLI quando cambia la libreria

## Problema

Un Jupyter Notebook e una CLI possono eseguire le stesse operazioni richiamando funzioni definite in una libreria comune.

Il problema nasce quando entrambi contengono direttamente una sequenza di chiamate:

```python
f1()
f2()
f3()
...
fn()
```

Se una funzione viene:

- rinominata;
- eliminata;
- aggiunta;
- riordinata;
- sostituita con più funzioni;

la sequenza presente nel Notebook e quella presente nella CLI possono divergere.

In questo caso la CLI non viene aggiornata automaticamente solo perché usa la stessa libreria: viene aggiornata automaticamente soltanto l'implementazione delle funzioni che continua a chiamare. L'orchestrazione rimane duplicata.

---

## Principio architetturale

Notebook e CLI non devono conoscere la sequenza completa delle funzioni interne.

Entrambi devono richiamare un unico entry point pubblico della libreria:

```text
Notebook ─┐
          ├──> API pubblica ──> orchestrazione ──> funzioni interne
CLI ──────┘
```

Da evitare:

```text
Notebook ──> f1, f2, f3, ..., fn
CLI ───────> f1, f2, f3, ..., fn
```

L'obiettivo è avere una sola definizione della pipeline.

---

## Struttura consigliata

```text
project/
├── mylib/
│   ├── __init__.py
│   ├── operations.py
│   └── pipeline.py
├── cli.py
└── analysis.ipynb
```

---

## Funzioni operative interne

```python
# mylib/operations.py

def load_data(config):
    ...

def validate_data(data, config):
    ...

def transform_data(data, config):
    ...

def generate_report(data, config):
    ...
```

Queste funzioni sono dettagli implementativi. Possono cambiare nome, essere sostituite o essere suddivise, purché venga aggiornata l'orchestrazione centrale.

---

## Orchestrazione centralizzata

```python
# mylib/pipeline.py

from .operations import (
    load_data,
    validate_data,
    transform_data,
    generate_report,
)


def run_pipeline(config):
    data = load_data(config)
    data = validate_data(data, config)
    data = transform_data(data, config)
    report = generate_report(data, config)

    return report
```

La funzione `run_pipeline()` rappresenta il contratto pubblico stabile.

Se viene aggiunto, rimosso o modificato uno step, la modifica viene effettuata soltanto qui.

---

## Uso dalla CLI

```python
# cli.py

from mylib import run_pipeline


def main():
    config = load_cli_configuration()
    result = run_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
```

La CLI deve essere un wrapper sottile. Deve occuparsi principalmente di:

- parsing degli argomenti;
- caricamento della configurazione;
- chiamata dell'entry point;
- formattazione dell'output;
- gestione del codice di uscita.

Non deve duplicare la logica applicativa.

---

## Uso dal Notebook

```python
from mylib import run_pipeline

result = run_pipeline(config)
result
```

In questo modo Notebook e CLI eseguono esattamente la stessa orchestrazione.

---

## Notebook con step separati

Se il Notebook deve mostrare risultati intermedi, è possibile usare una classe pipeline mantenendo comunque un unico metodo `run()`.

```python
class AnalysisPipeline:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.result = None

    def load(self):
        self.data = load_data(self.config)
        return self

    def validate(self):
        self.data = validate_data(self.data, self.config)
        return self

    def transform(self):
        self.data = transform_data(self.data, self.config)
        return self

    def generate(self):
        self.result = generate_report(self.data, self.config)
        return self

    def run(self):
        return (
            self.load()
            .validate()
            .transform()
            .generate()
        )
```

CLI:

```python
pipeline = AnalysisPipeline(config)
pipeline.run()
```

Notebook in modalità completa:

```python
pipeline = AnalysisPipeline(config)
pipeline.run()
```

Notebook in modalità esplorativa:

```python
pipeline = AnalysisPipeline(config)
pipeline.load()
pipeline.data
```

```python
pipeline.validate()
pipeline.data
```

```python
pipeline.transform()
pipeline.data
```

```python
pipeline.generate()
pipeline.result
```

La sequenza ufficiale continua a essere definita una sola volta nel metodo `run()`.

---

## API pubblica stabile

La libreria dovrebbe esporre soltanto gli oggetti che costituiscono il contratto pubblico.

```python
# mylib/__init__.py

from .pipeline import run_pipeline, AnalysisPipeline

__all__ = [
    "run_pipeline",
    "AnalysisPipeline",
]
```

Notebook e CLI dovrebbero importare:

```python
from mylib import run_pipeline
```

e non direttamente:

```python
from mylib.operations import transform_data
```

Questo permette di modificare liberamente i dettagli interni senza rompere i chiamanti.

---

## Pipeline dinamica

Se gli step devono essere configurabili, si può usare un registro.

```python
STEP_REGISTRY = {
    "load": load_data,
    "validate": validate_data,
    "transform": transform_data,
    "report": generate_report,
}
```

Configurazione:

```yaml
pipeline:
  - load
  - validate
  - transform
  - report
```

Esecuzione:

```python
def run_pipeline(config, context):
    for step_name in config["pipeline"]:
        try:
            step = STEP_REGISTRY[step_name]
        except KeyError as exc:
            raise ValueError(
                f"Step sconosciuto: {step_name}"
            ) from exc

        context = step(context)

    return context
```

Questa soluzione è utile quando serve:

- attivare o disattivare step;
- cambiare l'ordine;
- creare pipeline diverse;
- configurare l'esecuzione senza modificare il codice.

Il registro e i nomi simbolici diventano però parte del contratto e devono essere gestiti con attenzione.

---

## Rinominazione di funzioni pubbliche

Quando una funzione pubblica viene rinominata, conviene mantenere temporaneamente un alias compatibile.

```python
import warnings


def new_function_name(data):
    ...


def old_function_name(data):
    warnings.warn(
        "old_function_name è deprecata; usare new_function_name",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function_name(data)
```

Questo evita di rompere immediatamente:

- vecchi Notebook;
- versioni precedenti della CLI;
- script esterni;
- automazioni già distribuite.

---

## Versionamento

Per l'API pubblica è opportuno applicare il semantic versioning:

- **patch**: correzioni compatibili;
- **minor**: nuove funzionalità compatibili;
- **major**: modifiche incompatibili.

Le funzioni interne possono cambiare liberamente, purché il contratto pubblico resti stabile.

---

## Test necessari

La struttura architetturale riduce il rischio, ma la garanzia concreta deriva dai test.

### Test dell'entry point

```python
def test_pipeline_runs():
    result = run_pipeline(test_config)
    assert result is not None
```

### Test della CLI

```python
from typer.testing import CliRunner
from myproject.cli import app


def test_cli_runs():
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--config", "test.yml"],
    )

    assert result.exit_code == 0
```

### Test di equivalenza

```python
def test_cli_and_library_are_equivalent(tmp_path):
    expected = run_pipeline(test_config)

    actual = run_cli_and_read_output(
        test_config,
        tmp_path,
    )

    assert actual == expected
```

### Test del Notebook

Il Notebook può essere eseguito automaticamente in CI con strumenti come:

- `nbclient`;
- `nbconvert`;
- `pytest-notebook`.

Il test deve verificare che:

- tutte le celle vengano eseguite;
- non si verifichino eccezioni;
- gli output principali siano coerenti con quelli della libreria o della CLI.

---

## Evitare dipendenze dallo stato implicito del Notebook

Per mantenere l'equivalenza, il Notebook non dovrebbe dipendere da:

- variabili create manualmente in celle precedenti;
- celle eseguite fuori ordine;
- modifiche interattive non rappresentate nella configurazione;
- file temporanei non prodotti dalla pipeline;
- stato globale nascosto.

Tutti gli input necessari dovrebbero essere espliciti:

```python
config = {
    "input_path": "...",
    "output_path": "...",
    "option_a": True,
}

result = run_pipeline(config)
```

La stessa configurazione deve poter essere fornita alla CLI.

---

## Regole pratiche

1. Definire un unico entry point pubblico.
2. Centralizzare la sequenza degli step nella libreria.
3. Rendere la CLI un wrapper sottile.
4. Usare il Notebook come interfaccia alla stessa pipeline.
5. Evitare import diretti delle funzioni interne.
6. Separare API pubblica e dettagli implementativi.
7. Mantenere alias temporanei per le rinominazioni.
8. Testare libreria, CLI e Notebook.
9. Aggiungere un test esplicito di equivalenza.
10. Rendere configurazione, input e output riproducibili.

---

## Checklist di revisione

Prima di modificare la libreria verificare:

- [ ] Notebook e CLI chiamano lo stesso entry point.
- [ ] La sequenza degli step è definita in un solo punto.
- [ ] La CLI non contiene logica applicativa duplicata.
- [ ] Le funzioni interne non sono importate dai chiamanti.
- [ ] Le modifiche incompatibili sono versionate correttamente.
- [ ] Le rinominazioni pubbliche mantengono un alias deprecato.
- [ ] I test della pipeline passano.
- [ ] I test della CLI passano.
- [ ] Il Notebook viene eseguito automaticamente senza errori.
- [ ] Esiste un test di equivalenza sugli output.

---

## Sintesi

La CLI non rimane automaticamente equivalente al Notebook se entrambi contengono una propria sequenza di chiamate.

La soluzione consiste nel non sincronizzare due orchestrazioni separate, ma nell'eliminare la duplicazione:

```python
result = run_pipeline(config)
```

deve essere la chiamata comune usata sia dal Notebook sia dalla CLI.

L'orchestrazione appartiene alla libreria. Notebook e CLI sono soltanto due interfacce diverse verso lo stesso processo.
