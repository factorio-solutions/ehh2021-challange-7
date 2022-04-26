# Poznámky k nasazení dockeru Z-Case Plus
Je nutné nastavit naslouchání hostu na port (defaultně jsme dali 8501) a otevřít tento port do místní sítě.

Také nastavit správnou IP hostu, na které bude naslouchat. Teď je tam 0.0.0.0, takže všechny IP hostu, které má k dispozici.

Dále je potřeba namapovat externí `volume` na složku pro persistentní log.

```
version: "3.3"
services:
  zcaseplus:
    build: ""
    ports:
      - "0.0.0.0:8501:8501"
    image:  zcaseregistry.azurecr.io/zcaseplus:2022.1.0
    restart: always
    volumes:
      - F:/ZCASE_logs:/app/mnt/logs/
    deploy:
      resources:
        limits:
          cpus: '5'
          memory: 500M
```

## Troubleshooting
Když Apple 14. 4. 2022 skončil podporu mobility API, tak aplikace spadla při spuštění a bylo nutné odpárat ze souboru `utils/data_loader.py` sloupec s těmito daty a přetrénovat nový model.

## Lifecycle
Aplikace se spustí a stáhne si data z webu o počasí a mobilitách. Tato fáze musí proběhnout v pořádku napoprvé, později to umí běžet i z historických dat po odpojení od webu. Když se vrátí připojení, zase to naskočí a stáhne nová data (nemám to ale ověřené).

Pak to cyklicky dělá každých pět minut novou predikci pro hodinová okna.