# Dicionário de Dados — Spotify Tracks Dataset

- **Dataset:** Spotify Tracks Dataset (Maharshi Pandya)
- **URL Pública (Kaggle):** [https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **URL Pública de Download Direto (Hugging Face Mirror):** [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv)
- **Dimensões Brutas:** 114.000 registros (faixas musicais) e 21 colunas (incluindo o índice `Unnamed: 0`), abrangendo mais de 114 gêneros musicais distintos extraídos via Spotify Web API.

## Descrição das Variáveis (Colunas)

| Coluna | Tipo | Faixa / Valores | Descrição | Papel no Sistema |
| :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | Inteiro | `0` a `113999` | Índice sequencial gerado na exportação do CSV. | Removido na limpeza |
| `track_id` | String | ID alfanumérico | Identificador único da faixa na plataforma Spotify. | Chave de deduplicação |
| `artists` | String | Texto (sep. `;`) | Nome(s) do(s) artista(s) intérprete(s) da faixa. | Metadado / EDA |
| `album_name` | String | Texto | Nome do álbum ao qual a faixa pertence. | Metadado / EDA |
| `track_name` | String | Texto | Título oficial da faixa musical. | Metadado / EDA |
| `popularity` | Inteiro | `0` a `100` | Índice algorítmico de popularidade do Spotify, calculado com base no volume total e na recência das reproduções. | **Target 1 (Regressão)** e base para **`is_hit` (Target 2)** |
| `duration_ms` | Inteiro | `> 0` (ms) | Duração total da faixa em milissegundos. | Feature (Entrada) |
| `explicit` | Booleano | `True` / `False` | Indica se a letra da música contém conteúdo explícito (`True` = sim; `False` = não ou desconhecido). | Feature (Entrada) |
| `danceability` | Float | `0.0` a `1.0` | Mede o quão adequada a faixa é para dançar, combinando tempo, estabilidade rítmica, força da batida e regularidade. | Feature (Entrada) |
| `energy` | Float | `0.0` a `1.0` | Medida perceptual de intensidade e atividade. Faixas energéticas tendem a ser rápidas, altas e intensas. | Feature (Entrada) |
| `key` | Inteiro | `0` a `11` | Tonalidade musical estimada segundo notação padrão de classes de altura (`0` = Dó, `1` = Dó♯/Ré♭, ..., `11` = Si). | Feature (Entrada) |
| `loudness` | Float | `-60.0` a `0.0` dB | Volume médio geral da faixa em decibéis (dB), correlacionado à percepção física de amplitude. | Feature (Entrada) |
| `mode` | Binário | `0` ou `1` | Modalidade harmônica da faixa: `1` indica escala Maior e `0` indica escala Menor. | Feature (Entrada) |
| `speechiness` | Float | `0.0` a `1.0` | Detecta a presença de palavras faladas na gravação (valores próximos de `1.0` indicam podcasts/audiolivros; abaixo de `0.33`, predominância musical). | Feature (Entrada) |
| `acousticness` | Float | `0.0` a `1.0` | Medida de confiança de que a faixa seja predominantemente acústica (sem amplificação eletrônica pesada). | Feature (Entrada) |
| `instrumentalness` | Float | `0.0` a `1.0` | Estima a ausência de vocais na faixa. Valores acima de `0.5` sugerem faixas puramente instrumentais. | Feature (Entrada) |
| `liveness` | Float | `0.0` a `1.0` | Detecta a presença de público na gravação; valores superiores a `0.8` indicam alta probabilidade de gravação ao vivo. | Feature (Entrada) |
| `valence` | Float | `0.0` a `1.0` | Positividade musical transmitida pela faixa: valores altos evocam alegria/euforia; valores baixos, tristeza/melancolia. | Feature (Entrada) |
| `tempo` | Float | BPM (`> 0`) | Andamento estimado da faixa em batidas por minuto (BPM). | Feature (Entrada) |
| `time_signature` | Inteiro | `3` a `7` | Fórmula de compasso estimada, indicando quantas batidas existem por compasso (ex.: `4` = 4/4). | Feature (Entrada) |
| `track_genre` | String | 114 categorias | Gênero musical associado à faixa no catálogo do Spotify. | Feature na predição de popularidade e **Target 3 (Classificação Multiclasse)** |

## Variável Derivada (Engenharia de Atributos)

| Coluna Derivada | Tipo | Regra de Criação | Descrição |
| :--- | :--- | :--- | :--- |
| `is_hit` | Binário (`0`/`1`) | `1` se `popularity >= 50`, senão `0` | Variável-alvo binária criada para classificar faixas de alto desempenho comercial (*hits*) versus faixas de menor alcance. |
