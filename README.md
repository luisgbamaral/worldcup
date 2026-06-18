# worldcup — previsão de futebol internacional e da Copa de 2026

Modelos estatísticos que estimam a **probabilidade** de cada resultado de uma partida
(**1X2** = vitória do mandante / empate / vitória do visitante) e, juntando isso numa
simulação do torneio, a chance de cada seleção ser campeã em 2026.

> **Resultado prático (ao vivo, junho/2026):** 🇪🇸 Espanha favorita (~62%), 🇦🇷 Argentina vice (~28%).
> **Resultado de pesquisa:** modelos *foundation* tune-free igualam ou superam os modelos
> tradicionais — e vencem com folga **quando há pouco dado de treino**.

---

### As métricas (todas: *quanto menor, melhor*, exceto onde indicado)
Avaliamos **probabilidades**, não só "acertou/errou". As regras abaixo são *proper scoring
rules*: só dão a melhor nota se o modelo reporta a probabilidade honesta.

- **RPS** (*Ranked Probability Score*) — **métrica principal.** É a régua certa para resultados
  **ordenados**: vitória → empate → derrota têm ordem natural. Mede o quão longe a probabilidade
  *acumulada* prevista ficou da realidade. Errar prevendo "empate" quando deu "vitória do
  mandante" custa menos do que prever "derrota" — porque está mais perto na ordem.
- **Log-loss** — o `−log` da probabilidade dada ao resultado que de fato ocorreu. Pune com força
  o modelo **confiante e errado**.
- **Brier** — erro quadrático médio entre as probabilidades previstas e o resultado (0/1). É o
  análogo do MSE para classificação.
- **ECE** (*Expected Calibration Error*) — mede **calibração**: quando o modelo diz "60%", o evento
  acontece ~60% das vezes? É a distância entre confiança e frequência real.

### Os modelos
- **Regressão logística (LogReg)** — a base conhecida: linear nas variáveis, devolve a
  probabilidade de cada resultado. É a **referência** contra a qual tudo é comparado.
- **Boosting — XGBoost, CatBoost** — um *comitê de árvores de decisão* construído **em sequência**:
  cada nova árvore corrige o **erro (resíduo)** das anteriores. É o estado da arte para dados
  tabulares, mas **exige calibrar hiperparâmetros** (profundidade, taxa de aprendizado, etc.).
- **Foundation models tabulares — TabPFN, TabICL** — redes pré-treinadas (a mesma família de
  *transformers* dos LLMs), treinadas em **milhões de tabelas sintéticas** para aprender "como
  prever a partir de uma tabela". Você entrega as linhas de treino e elas preveem linhas novas
  **numa única passada, sem treinar um modelo novo nem ajustar hiperparâmetro** — uso *zero-shot*,
  análogo a usar um modelo pré-treinado direto. **A pergunta de pesquisa:** esses modelos
  *tune-free* alcançam os clássicos cuidadosamente tunados?
- **Elo** — o *rating* de força das seleções (como no xadrez), atualizado a cada jogo. Aqui ele é a
  **variável explicativa** mais importante (uma covariável), **nunca um modelo concorrente**.

---

## O achado central

Com **bastante** histórico de treino, tudo empata: uma regressão logística é tão boa quanto
boosting ou foundation models (nada vence a logística de forma estatisticamente significativa).
Mas com **pouco** dado e **muitas** variáveis — o regime de uma Copa — os **foundation models
vencem com significância e a custo zero de ajuste**, enquanto os clássicos precisam de anos de
dados para empatar. Essa **eficiência de dados** é a contribuição do trabalho. Detalhes e tabelas
em [`reports/RESULTS.md`](reports/RESULTS.md) e [`paper/short_paper.md`](paper/short_paper.md).

## Como o repositório se organiza
```
src/worldcup/   o "motor": dados, Elo, features (107, sem vazamento de informação), modelos
experiments/    estudos de avaliação e ablação (a base do paper)
production/      previsor da Copa de 2026 (roda sozinho)
reports/         saídas geradas: gráficos e tabelas
paper/           o texto curto (short paper)
data/raw/        dados de origem — somente leitura (atualizados por um script, nunca à mão)
```
Convenções, decisões e regras de uso completas estão em [`CLAUDE.md`](CLAUDE.md).

## Como rodar
```bash
pip install -r requirements.txt          # Python >= 3.10
pytest                                    # suíte de testes
python -m worldcup.update                 # baixa os dados públicos mais recentes
python production/predict_wc2026.py       # previsão do campeão de 2026
```
O TabPFN usa uma API de nuvem gratuita: ponha o token em `.env` (modelo em `.env.example`).

## Fontes de dados
Resultados internacionais (martj42), grupos/calendário (openfootball) e dados de elenco —
todos públicos, baixados por `python -m worldcup.update`.
