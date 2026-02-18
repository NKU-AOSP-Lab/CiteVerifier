# Configuration

## Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | DBLP data directory |
| `DB_PATH` | `${DATA_DIR}/dblp.sqlite` | SQLite path for queries |
| `CORS_ORIGINS` | `http://localhost:8090` | Allowed frontend origins |
| `DBLP_XML_GZ_URL` | `https://dblp.org/xml/dblp.xml.gz` | DBLP source URL |
| `DBLP_DTD_URL` | `https://dblp.org/xml/dblp.dtd` | DTD source URL |
| `BATCH_SIZE` | `1000` | Build pipeline batch size |
| `PROGRESS_EVERY` | `10000` | Progress report interval |

## Data Files

Typical files under `DATA_DIR`:

- `dblp.xml.gz`
- `dblp.xml`
- `dblp.dtd`
- `dblp.sqlite`
