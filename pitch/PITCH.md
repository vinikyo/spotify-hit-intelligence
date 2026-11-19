# Pitch do Projeto — Spotify Hit & Genre Intelligence

- **Apresentação Interativa Pública (Slides Web):** [https://vinikyo.github.io/spotify-hit-intelligence/pitch/](https://vinikyo.github.io/spotify-hit-intelligence/pitch/)
- **Dupla:** Vinícius Francisco Garcia Sobral (`cg3034241`) e Arthur de Morais Marques (`cg3033856`)
- **Repositório GitHub:** [https://github.com/vinikyo/spotify-hit-intelligence](https://github.com/vinikyo/spotify-hit-intelligence)

---

## Roteiro de Apresentação (3 a 5 minutos)

### Slide 1 — Capa
> Olá! Somos Vinícius Sobral e Arthur Marques, e apresentamos o **Spotify Hit & Genre Intelligence**: um sistema de Ciência de Dados e Aprendizado de Máquina criado para prever a popularidade de músicas, detectar potenciais *hits* e classificar gêneros musicais automaticamente a partir de atributos acústicos do Spotify.

### Slide 2 — O Problema
> Hoje, mais de 100 mil novas faixas são publicadas todos os dias nas plataformas de *streaming*. Na nossa base de dados, **10,4% das faixas têm popularidade zero** e apenas **23,6% alcançam popularidade maior ou igual a 50**. Gravadoras, curadores e artistas independentes ainda dependem muito de intuição para decidir onde investir.

### Slide 3 — A Solução
> Desenvolvemos um pipeline reproduzível que transforma características objetivas do áudio em três respostas práticas:
> 1. **Escore Contínuo de Popularidade (0–100)** via regressão;
> 2. **Radar de Hits (`popularity >= 50`)** via classificação binária;
> 3. **Auto-Tagging de Gênero** usando exclusivamente atributos acústicos.

### Slide 4 — Dataset e Limpeza Rigorosa
> Utilizamos o *Spotify Tracks Dataset* (Maharshi Pandya, Kaggle), com **114.000 registros** e **114 gêneros**. Um cuidado técnico essencial foi remover **24.260 duplicatas de `track_id`**, garantindo **89.740 faixas únicas** sem vazamento de dados entre treino e teste.

### Slide 5 — Descobertas da Análise Exploratória
> Correlações lineares isoladas são fracas (por exemplo, `instrumentalness` tem correlação de `-0,13` e `loudness` de `+0,07`), mas o desempenho varia drasticamente entre gêneros (como `k-pop` com média `59,2` versus nichos abaixo de `5`). Isso provou que precisávamos de modelos não lineares.

### Slide 6 — Objetivos (Targets) e Modelagem
> Estruturamos 3 targets bem definidos com validação 80/20 estratificada, comparando modelos lineares (*Ridge* / Regressão Logística) contra ensembles baseados em árvores (*Random Forest* e *HistGradientBoosting*).

### Slide 7 — Resultados do Radar de Hits (`is_hit`)
> O **HistGradientBoosting** alcançou **ROC-AUC de 0,865**, **77,7% de acurácia** e **78,7% de revocação**, identificando quase 8 em cada 10 *hits* reais no conjunto de teste.

### Slide 8 — Explicabilidade e Demais Targets
> - Na **Regressão (`popularity`)**, o *Random Forest* atingiu **$R^2 = 0{,}4394$** e **RMSE = 15,32** (+11,1 p.p. acima do modelo linear).
> - Na **Classificação de 10 Gêneros** usando *somente áudio*, o *Random Forest* obteve **66,03% de acurácia** — mais de 6,6 vezes acima do acaso.
> - Os fatores mais relevantes foram `track_genre`, `loudness`, `duration_ms`, `valence`, `energy` e `acousticness`.

### Slide 9 — Impacto e Próximos Passos
> O sistema apoia decisões de A&R, artistas independentes e curadores de playlists. Como evolução, planejamos incorporar histórico de seguidores do artista, tendências virais e redes neurais sobre espectrogramas de áudio.

### Slide 10 — Encerramento
> Todo o projeto — código Python, métricas reais, artigo científico, dicionário de dados e slides — está público no GitHub. Obrigado!
