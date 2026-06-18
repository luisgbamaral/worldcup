# CLAUDE.md — worldcup

Memória de projeto do Claude Code. Versionada no repositório: quem clona já tem o contexto
completo, **sem briefing verbal**. Preferências pessoais ficam em
`~/.claude/CLAUDE.md`; em conflito, este arquivo de projeto vence.

> **Idioma:** converse e escreva este arquivo em português; mantenha **todo o código em inglês**
> (nomes, comentários, docstrings, logs, mensagens de commit). Termos técnicos, caminhos e
> comandos não se traduzem.

## Como responder
- Aja quando tiver informação suficiente. Dê uma recomendação, não um cardápio de opções.
- Faça a alteração mínima que resolve a tarefa; não reescreva um arquivo inteiro por um conserto
  pontual. Mostre o diff relevante, não o arquivo todo.
- Reuse o que já existe em `src/worldcup/` antes de criar função nova; prefira funções pequenas a
  abstrações genéricas. *Over-engineering* aqui é concreto — ex.: para somar uma métrica a uma
  tabela, escreva `def log_loss_by_model(probs): ...` e chame direto; **não** crie uma
  `class MetricRegistry` com factory e config plugável. Se ~10 linhas resolvem, não faça uma classe.
- **Relate com fidelidade.** Se um resultado deu nulo, um teste falhou ou um passo foi pulado,
  diga isso. Nunca ajuste nem selecione resultados para um achado parecer melhor do que é.

## Objetivo
Previsão de futebol internacional, com a **Copa de 2026** como aplicação principal. Três fins:
(1) ganhar um bolão; (2) um paper curto para periódico; (3) portfólio sênior de ML.

## Equipe e responsabilidades
- **Luís Guilherme Brandão Amaral** (LEME) — autor/lead; modelagem, features e avaliação.
- **Claude (Opus)** — par de programação; mensagens de commit terminam com `Co-Authored-By: Claude ...`.
- Repositório solo hoje; estas convenções mantêm o onboarding pronto para novos membros.

## Estrutura — pastas e função
```
src/worldcup/     pacote instalável = o MOTOR (mantenha estável, reusado em todo lugar)
  config.py  data.py  clean.py  lookups.py   — caminhos, carga, nomes canônicos de seleções
  elo.py  ratings.py                         — Elo próprio + sistemas de rating
  features.py                                — 107 features sem leakage (+ cache)
  modeling.py                                — classificadores, calibração, RPS, CV temporal, HPO
  update.py                                  — atualiza dados crus das fontes públicas
experiments/      scripts de pesquisa (ablations, avaliações) — o PAPER nasce daqui
production/       previsor 2026 standalone (depende só de src/, NÃO do paper)
paper/            short_paper.md — o texto
reports/          saída GERADA: figures/ e tables/ (versionadas)
scripts/          entry points de build/update
tests/            suíte pytest
data/             ver proteções e camadas geradas abaixo
notebooks/  models/
```

## Proteções — não editar
A aplicação destas regras mora no `.claude/settings.json` (deny); este arquivo só as descreve.
- **`data/raw/**`** — dados crus de origem. Nunca edite à mão. Atualize **só** via
  `python -m worldcup.update` (baixa das fontes canônicas no GitHub). Trate como entrada read-only.
- **IMPORTANT — `.env`**: guarda `TABPFN_TOKEN`. Nunca leia para o contexto, nunca commite (está
  no gitignore). Ler um segredo para o contexto é irreversível.
- Não dê **push em `main`** nem rode `worldcup.update` sem confirmar antes.

## Onde vai a saída gerada
- `reports/figures/`, `reports/tables/` — artefatos versionados (gráficos, tabelas LaTeX/CSV).
- `data/processed/`, `data/interim/` — camadas geradas, gitignored. Pode apagar e regerar.
- `production/outputs/` — CSVs de previsão, gitignored.

## Estilo de código
- Use `snake_case` em funções/variáveis, `UPPER_SNAKE` em constantes de módulo, `PascalCase` em
  classes. Mantenha nomes de módulo curtos e minúsculos.
- Use Polars para dataframes (não pandas), a menos que uma dependência force o contrário.
- Comece cada módulo com uma docstring de uma linha dizendo o que faz e como rodar. Comente o
  *porquê*, não o *o quê*. Acompanhe a densidade do código ao redor.
- Escreva logs em inglês, enxutos e append-only (ex.: `print(..., flush=True)` em runs longos).

## Stack e como rodar
- **Python ≥ 3.10** (esta máquina roda 3.13). Só Python — sem R nem Julia.
- Libs centrais: Polars, NumPy, scikit-learn, XGBoost, CatBoost, Optuna, Matplotlib, PyArrow.
  Foundation models: TabPFN (nuvem, precisa de `TABPFN_TOKEN`) e TabICL (CPU, local).
- Segredos em `.env` (gitignored); template em `.env.example`.

```bash
pytest                                   # suíte de testes (pythonpath=src no pyproject)
python -m worldcup.update                # atualiza data/raw/ das fontes públicas
python scripts/build_features.py         # reconstrói o cache de features
python experiments/<nome>.py             # um experimento / ablation
python production/predict_wc2026.py      # a previsão do campeão 2026
```
Experimentos pesados (foundation models em CPU, Optuna) levam de minutos a horas: prefira rodar em
background com log append-only.

## Decisões arquiteturais — não reverter em silêncio
- **Elo é COVARIÁVEL, nunca baseline.** Nenhuma linha "Elo" em benchmark; a referência de
  significância é o modelo de features mais simples (LogReg em todas as features).
- **Sem leakage e estritamente temporal.** Toda feature usa só dados anteriores à partida; a CV é
  walk-forward, **nunca** k-fold embaralhado; calibração em fatia temporal.
- **RPS é a métrica primária** (1X2 ordenado); log-loss/Brier/ECE são secundárias.
- **Foundation models ficam tune-free** (TabPFN/TabICL) — essa propriedade configuration-free é a
  alegação de pesquisa sob teste; só baselines clássicos podem ser tunados.
- Reuse o cache de features (`features.build_match_features_cached`); o build linha-a-linha é caro.

## Exemplo de tarefa bem-formada (Contexto → Exemplo → Plano → Escopo)
> **Contexto:** `experiments/worldcup_eval.py` já faz avaliação 1X2 one-step com RPS + testes de Holm.
> **Quero:** adicionar log-loss multiclasse ao lado do RPS nas tabelas de série e de Copa,
> calculado a partir das probabilidades por partida já salvas — mesmo estilo de
> `reports/tables/onestep_series.tex`.
> **Plano:** ler `data/processed/onestep_*.parquet`; calcular log-loss por modelo; anexar a coluna;
> reemitir o LaTeX via `viz.df_to_neurips_latex`.
> **Escopo:** não retreine modelos, não toque em `data/raw/`, mantenha o RPS como métrica primária.
```