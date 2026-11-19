# Entrega da Atividade — Spotify Hit & Genre Intelligence

## 1. Identificação da Dupla e Links Públicos

- **Integrante 1:** Vinícius Francisco Garcia Sobral — **Prontuário:** `cg3034241`
- **Integrante 2:** Arthur de Morais Marques — **Prontuário:** `cg3033856`
- **Repositório GitHub (Público):** [https://github.com/vinikyo/spotify-hit-intelligence](https://github.com/vinikyo/spotify-hit-intelligence)
- **Pitch do Projeto (Link Público Interativo):** [https://vinikyo.github.io/spotify-hit-intelligence/pitch/](https://vinikyo.github.io/spotify-hit-intelligence/pitch/) (também disponível em `pitch/index.html` e `pitch/PITCH.md`)
- **Artigo Científico (Link Público / PDF):** [https://vinikyo.github.io/spotify-hit-intelligence/artigo/](https://vinikyo.github.io/spotify-hit-intelligence/artigo/) (também disponível em `artigo/artigo_cientifico.md` e `artigo/index.html`)

---

## 2. Conjunto de Dados (*Dataset*)

- **Nome do Dataset:** Spotify Tracks Dataset (Maharshi Pandya)
- **URL Pública (Kaggle):** [https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **URL Pública de Download Direto (Espelho Hugging Face):** [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv)
- **Volume:** 114.000 registros brutos (89.740 faixas únicas após deduplicação por `track_id`), 20 colunas úteis e 114 gêneros musicais.

### Dicionário de Dados (Principais Variáveis)

| Variável | Tipo | Descrição Breve |
| :--- | :--- | :--- |
| `track_id` | String | Identificador único da faixa no Spotify (usado para deduplicação). |
| `artists` / `album_name` / `track_name` | String | Metadados textuais de artista(s), álbum e título da faixa. |
| `popularity` | Inteiro (`0–100`) | Índice algorítmico de popularidade baseado em volume e recência de reproduções. |
| `duration_ms` | Inteiro | Duração da faixa em milissegundos. |
| `explicit` | Booleano | Indica presença de conteúdo explícito na letra (`True`/`False`). |
| `danceability` | Float (`0–1`) | Adequação da música para dançar (ritmo, regularidade e força da batida). |
| `energy` | Float (`0–1`) | Medida perceptual de intensidade, atividade e volume sonoro. |
| `key` / `mode` | Inteiro / Binário | Tonalidade musical (`0–11`) e modalidade harmônica (`1` = Maior, `0` = Menor). |
| `loudness` | Float (dB) | Volume médio geral da faixa em decibéis (tipicamente `-60` a `0` dB). |
| `speechiness` | Float (`0–1`) | Presença de palavras faladas na gravação. |
| `acousticness` | Float (`0–1`) | Grau de confiança de que a faixa seja predominantemente acústica. |
| `instrumentalness` | Float (`0–1`) | Probabilidade de ausência de vocais na faixa. |
| `liveness` | Float (`0–1`) | Probabilidade de presença de audiência (gravação ao vivo). |
| `valence` | Float (`0–1`) | Positividade musical transmitida (alegre/eufórica vs. melancólica). |
| `tempo` / `time_signature` | Float / Inteiro | Andamento em BPM e fórmula de compasso estimada (`3` a `7`). |
| `track_genre` | Categórica | Gênero musical atribuído à faixa no catálogo (114 categorias). |
| `is_hit` *(derivada)* | Binária (`0`/`1`) | Variável derivada: `1` se `popularity >= 50` (*Hit*), `0` caso contrário. |

> Dicionário detalhado disponível em [`docs/dicionario_dados.md`](dicionario_dados.md).

---

## 3. Objetivos do Sistema (*Targets*)

O objetivo principal do sistema é **prever o potencial comercial de faixas musicais e classificar automaticamente seu gênero a partir de atributos acústicos extraídos da Spotify Web API**, fornecendo inteligência analítica para gravadoras, produtores, curadores de playlists e artistas independentes. Para isso, definiram-se três variáveis-alvo (*targets*):

1. **Target 1 — Regressão (`popularity`):** Estimar continuamente o escore de popularidade (`0` a `100`) de uma faixa a partir de seus atributos acústicos e gênero musical (melhor resultado: *Random Forest Regressor* com $R^2 = 0{,}4394$ e $\text{RMSE} = 15{,}316$).
2. **Target 2 — Classificação Binária (`is_hit`):** Classificar se uma música alcançará alto desempenho comercial (`popularity >= 50`, correspondente a 23,63% do catálogo único) (melhor resultado: *HistGradientBoosting* com $\text{ROC-AUC} = 0{,}8649$, Acurácia de $77{,}67\%$ e Revocação de $78{,}69\%$).
3. **Target 3 — Classificação Multiclasse (`track_genre`):** Reconhecer automaticamente o gênero musical entre 10 grandes estilos utilizando **exclusivamente** atributos de áudio (melhor resultado: *Random Forest* com Acurácia de $66{,}03\%$ e $\text{F1-Macro} = 0{,}6632$).

---

## 4. Descrição do Sistema (Máximo de 200 Palavras)

O **Spotify Hit & Genre Intelligence** é um sistema analítico e preditivo baseado em Aprendizado de Máquina, desenvolvido para apoiar gravadoras, produtores musicais, curadores de playlists e artistas independentes na tomada de decisão orientada por dados. A partir do *Spotify Tracks Dataset* (114.000 faixas e atributos acústicos extraídos via Spotify Web API, como dançabilidade, energia, valência, volume, andamento e duração), o sistema executa um pipeline automatizado de pré-processamento, análise exploratória e modelagem supervisionada. Seu funcionamento organiza-se em três frentes complementares: (1) regressão para estimar o índice contínuo de popularidade (`popularity`, de 0 a 100); (2) classificação binária para identificar faixas com alto potencial comercial (*hits*, com `popularity >= 50`); e (3) classificação multiclasse para reconhecer automaticamente o gênero musical (`track_genre`) apenas pelas características sonoras. Os algoritmos avaliados — Regressão Linear/Logística, Random Forest e HistGradientBoosting — são comparados por métricas padronizadas (RMSE, MAE, R², F1-Score e ROC-AUC), gerando diagnósticos visuais de correlação e importância de variáveis. Dessa forma, o sistema traduz sinais acústicos complexos em recomendações práticas para curadoria, catalogação automática e estratégia de lançamento musical.

*(Contagem: 175 palavras — dentro do limite máximo de 200 palavras.)*
