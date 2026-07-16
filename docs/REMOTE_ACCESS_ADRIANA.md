# Accesso remoto ad Adriana via tunnel SSH inverso

**Problema**: Adriana è dietro un firewall aziendale che blocca le connessioni
in entrata. Non è possibile collegarsi direttamente via SSH da remoto (es. da
Irina o da qualsiasi altra macchina fuori dalla rete aziendale).

**Soluzione**: tunnel SSH inverso permanente verso la VPS `tslab.investia.cloud`
(IP pubblico, già nell'infrastruttura del progetto). Adriana apre una
connessione SSH *in uscita* verso la VPS (le connessioni in uscita non sono
bloccate dal firewall aziendale) e mantiene aperto un tunnel inverso. Da quel
momento, chiunque ha accesso SSH alla VPS può raggiungere Adriana passando
attraverso di essa come jump host.

`autossh` garantisce che il tunnel si riconnetta automaticamente se cade
(rete instabile, riavvio della VPS, ecc.) — gestito come servizio `systemd`
che parte da solo al boot.

---

## 1. Setup su Adriana

Installa autossh:
```bash
sudo apt install autossh
```

Genera una chiave SSH dedicata al tunnel, senza passphrase (necessaria per
l'avvio automatico non interattivo del servizio):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_tunnel -N ""
ssh-copy-id -i ~/.ssh/id_tunnel.pub luca@tslab.investia.cloud
```

Crea il servizio systemd:
```bash
sudo nano /etc/systemd/system/tunnel-adriana.service
```

Contenuto:
```ini
[Unit]
Description=Tunnel SSH inverso verso VPS (accesso remoto Adriana)
After=network-online.target
Wants=network-online.target

[Service]
User=luca
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/bin/autossh -M 0 -N \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    -o "ExitOnForwardFailure yes" \
    -i /home/luca/.ssh/id_tunnel \
    -R 2222:localhost:22 \
    luca@tslab.investia.cloud
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Attiva e avvia:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tunnel-adriana.service
sudo systemctl status tunnel-adriana.service
```

---

## 2. Setup sulla VPS (tslab.investia.cloud)

Verifica che il forwarding TCP sia abilitato:
```bash
sudo grep -i "GatewayPorts\|AllowTcpForwarding" /etc/ssh/sshd_config
```

Se mancano o sono impostati su `no`, aggiungi/modifica in `/etc/ssh/sshd_config`:
```
AllowTcpForwarding yes
GatewayPorts no
```

> `GatewayPorts no` è la configurazione corretta e sicura: la porta 2222
> resta accessibile solo da `localhost` sulla VPS stessa, cioè solo passando
> per un jump host SSH autenticato — non viene esposta a tutto internet.

Riavvia il servizio SSH:
```bash
sudo systemctl restart sshd
```

---

## 3. Connessione da una macchina remota (es. Irina)

Connessione diretta:
```bash
ssh -J luca@tslab.investia.cloud luca@localhost -p 2222
```

Oppure, per comodità, aggiungi un alias in `~/.ssh/config` sulla macchina da
cui ti connetti:
```
Host adriana
    HostName localhost
    Port 2222
    User luca
    ProxyJump luca@tslab.investia.cloud
```

Da quel momento basta:
```bash
ssh adriana
```

---

## 4. Verifica stato del tunnel

Sulla VPS, controlla che il tunnel sia attivo:
```bash
ss -tlnp | grep 2222
```
Deve mostrare `127.0.0.1:2222` in stato LISTEN.

Su Adriana, controlla lo stato del servizio:
```bash
sudo systemctl status tunnel-adriana.service
```

---

## Note

- Nessuna modifica al firewall aziendale è necessaria: il tunnel si basa
  solo su connessioni in uscita da Adriana, normalmente permesse.
- Se in futuro serve lo stesso accesso anche per altre macchine dietro
  firewall (es. eventuali postazioni aggiuntive), si replica lo stesso
  schema usando una porta diversa da 2222 per ciascuna macchina
  (es. 2223, 2224, ...) sulla stessa VPS.
