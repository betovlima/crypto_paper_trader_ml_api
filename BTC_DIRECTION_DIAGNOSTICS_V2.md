# BTC Direction & Return V2: corrigir e diagnosticar a regressão

## Por que esta versão existe

A primeira auditoria com quatro anos de Binance Spot identificou regressões
de duas horas inteiramente constantes em três folds. Os classificadores de
duas horas mostraram ROC-AUC pouco acima de 0,5, mas a vantagem contra a
frequência de altas do treino foi pequena.

Esta versão **não** procura automaticamente rentabilidade. Ela investiga
se os modelos aprendem uma relação preditiva, mantendo o modo Target/Stop
e a execução V1 como referência. Não integra com ordens Live.

## Regressão em pontos-base e parâmetros independentes

Retorno simples real:

`r(t,h) = close(t+h) / close(t) - 1`

Alvo do regressor:

`y_scaled = 10000 * r(t,h)`

Predição para auditoria:

`predicted_return = predicted_scaled / 10000`

Assim, 0,5% de retorno corresponde a 50 pontos-base no treinamento,
mas volta a 0,005 nos resultados. Os valores de 1, 100 e 10000 podem ser
avaliados controladamente usando `forecast.regression_target_scale`.

Os campos novos ficam na tabela `ml_runtime_parameters` e na tela
**Parâmetros**, sem `.env` e sem substituir opções já personalizadas.

| Chave de parâmetro | Padrão | Significado |
| --- | ---: | --- |
| forecast.regression_target_scale | 10000 | Fator aplicado ao alvo no ajuste |
| forecast.regression_n_estimators | 350 | Máximo de árvores |
| forecast.regression_max_depth | 4 | Profundidade das árvores |
| forecast.regression_learning_rate | 0,04 | Taxa de aprendizado |
| forecast.regression_min_child_weight | 4 | Mínimo de Hessian acumulado para divisão |
| forecast.regression_gamma | 0 | Penalidade mínima de ganho para divisão |
| forecast.regression_reg_alpha | 0 | Penalidade L1 |
| forecast.regression_reg_lambda | 1,5 | Penalidade L2 |
| forecast.regression_early_stopping_rounds | 30 | Paciência de parada |
| forecast.bootstrap_samples | 200 | Reamostragens por blocos temporais |

O classificador continua utilizando os parâmetros `xgboost.*`.
O regressor ainda compartilha configurações de infraestrutura,
aleatoriedade e amostragem `xgboost.tree_method`,
`xgboost.random_state`, `xgboost.n_jobs`,
`xgboost.subsample` e `xgboost.colsample_bytree`.
Os parâmetros que controlam divisão, escala, regularização e parada
do regressor foram separados.

## Novos diagnósticos

Cada fold registra dentro de `folds.json` e de
`forecast_metrics.json`:

- Classificação: probabilidade bruta OOS e calibrada OOS, Brier antes
  e depois, referência do treino, coeficiente/intercepto da calibração
  e tabela de calibração em 10 faixas.
- XGBoost: número total de árvores, `best_iteration`,
  árvores efetivamente utilizadas na previsão, nós de divisão,
  folhas e árvores que nunca dividiram.
- Regressão: número de retornos distintos e desvio-padrão das
  previsões, com unidade final em retorno simples.
- Generalização: intervalos percentis de 95% de ROC-AUC, Brier Skill
  e melhora de MAE contra zero. Reamostragens cronológicas em blocos,
  estratificadas por fold, usando bloco de pelo menos 1 dia ou duas
  vezes o horizonte futuro.

Os intervalos consideram parte da dependência serial, mas não
eliminam mudança de regime, sobreajuste ou viés de escolha.

As métricas de teste não participam do treinamento nem da escolha
da calibração.

## Comparação controlada com o primeiro ZIP Binance

Ao executar uma nova coleta, o início e o fim podem mudar
de acordo com o calendário. Para não comparar períodos diferentes,
o comando **replay** usa os candles do ZIP anterior e preserva
os parâmetros de classificação, features, validação e folds.

Na pasta da ML API:

```bash
git fetch origin
git switch --track origin/feature/btc-direction-diagnostics-v2-20261108
poetry run python -m pytest tests/test_forecast_diagnostics.py tests/test_forecast_comparison.py tests/test_forecast_replay.py -v
poetry run python -m pytest -v
```

Para reproduzir a V2 a partir de uma auditoria original (coloque o ZIP
na pasta do projeto ou informe o caminho real do arquivo):

```bash
poetry run python scripts/replay_btc_forecast.py "btc-forecast-audit-558e5d65-0c92-42a7-9f9b-6f15f197af47.zip"
```

O programa gera um arquivo `-v2-replay.zip` no mesmo diretório
do original. Ele também verifica:

1. Fonte Binance, símbolo e timeframes idênticos.
2. Checksum do conteúdo dos históricos brutos, decisão e tendência.
3. Exatamente as mesmas fronteiras de treino, validação, calibração
   e teste, ignorando apenas novos campos de diagnóstico.
4. A mesma sequência de retornos futuros realizados em cada horizonte.

Se qualquer checagem falhar, a comparação é recusada. O replay
não baixa dados de mercado, e o novo ZIP contém o SHA-256 da
auditoria de origem. Não há fallback entre corretoras.

Para comparar manualmente duas auditorias com os mesmos candles:

```bash
poetry run python scripts/compare_forecast_audits.py "V1.zip" "V2.zip"
```

## O que testar

Primeiro comparar os valores por fold:

- `split_nodes_used` passou a ser maior que zero?
- `unique_predictions` da regressão de 2 e 6 horas aumentou?
- As previsões de retorno passaram a superar retorno zero e
  média histórica no período OOS?
- A probabilidade calibrada melhorou contra a probabilidade bruta
  e o baseline do treino?
- Os intervalos de confiança excluem vantagem nula, ou o
  sinal é apenas ruído temporal?

**Não executar LHS ou CARO ainda:** só depois de confirmar
modelos não degenerados e uma vantagem fora da amostra.
