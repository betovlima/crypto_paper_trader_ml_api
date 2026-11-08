# BTC Forecast: experimento de direção e retorno · Etapa 1

## Objetivo

Separar o problema **prever o movimento do BTC** da antiga estratégia que
prevê o atingimento de alvos e stops e depois tenta operar BTC/CASH.

As duas experiências coexistem na mesma ML API e possuem IDs, parâmetros
e ZIPs independentes. O padrão permanece `TARGET_STOP` e a main não
foi modificada. Não existem ordens reais nem simuladas neste novo modo.

## Seleção na tela

Na aba **ML Backtest → Nova execução**, escolher:

- **Tipo de experimento:** `BTC Direction & Return`
- **Fonte histórica:** `Binance Spot Archive`
- **Símbolo:** `BTCUSDT`
- **Período:** `4 anos`
- **Candle de decisão:** `30min`
- **Candle de tendência:** `4hour`

As opções `backtest.default_experiment_type`, `forecast.horizons_hours`
e `forecast.maximum_folds` ficam no banco e podem ser ajustadas
no painel **Parâmetros**. Valores iniciais: modo legado, horizontes
de 2/6/24 horas e três folds no modelo direto. Não existem parâmetros
`.env` de negociação para este experimento.

## Definição dos alvos

Para o candle que fechou em `t`, e horizonte `h`, definimos:

`future_return_h = close[t+h] / close[t] - 1`

`future_up_h = 1 se future_return_h > 0; caso contrário 0`.

O alvo é o fechamento futuro, **não** o maior preço futuro.
Cada horizonte tem modelos de classificação e regressão separados.

Os alvos de 2, 6 e 24 horas equivalem, em 30 minutos, a
4, 12 e 48 candles futuros, respectivamente.

Se o horizonte não couber integralmente antes de uma interrupção
de mercado documentada, o rótulo é inválido e não entra no treinamento
ou na avaliação. Preços faltantes nunca são interpolados.

## Método de avaliação

1. Engenharia de atributos com indicadores de timeframes maiores disponíveis
   somente depois do fechamento dos candles correspondentes.
2. Treino XGBoost para classificação da direção e regressão do retorno.
3. Separação cronológica de quatro partições: treino, validação para
   early stopping, calibração de probabilidade e teste fora da amostra.
4. Expurgo entre as partições por horizonte `h` e verificação adicional
   do timestamp em que o alvo já seria conhecido.
5. Testes walk-forward com número de folds definido no banco e parâmetros
   congelados durante cada execução.
6. Baselines construídos apenas com o **treino**: frequência histórica
   de alta e retorno médio; também comparados com retorno futuro zero.

A calibração é ajustada depois do early stopping, em uma partição
diferente. Isso reduz reutilização inadequada dos rótulos.

## Indicadores principais de avaliação

| Tarefa | Modelo | Referência |
| --- | --- | --- |
| Alta/baixa | ROC-AUC, PR-AUC, Log Loss, Brier | Frequência de alta do treino |
| Qualidade de probabilidades | Brier Skill | Brier de frequência do treino |
| Retorno percentual | MAE e RMSE | Retorno zero e retorno médio do treino |
| Generalização | Métricas por fold | Fronteiras de cada período |

**Brier Skill positivo não garante rentabilidade.** A execução V1 não
avalia uma política de capital ou decisões BTC/CASH.

## Auditoria da execução

O botão **Exportar auditoria ZIP**, na aba BTC Forecast, exporta
arquivos completos e não reduzidos para exibição:

- `manifest.json`: tipo, origem, checksums e horizontes
- `forecast_predictions_oos.csv`: todas as previsões, retornos
  realizados, probabilidades, folds e atributos usados
- `forecast_metrics.json`: métricas por horizonte e por fold
- `folds.json`: datas, dimensões e expurgos
- `history_coverage.json` e `market_interruptions.json`
- `parameters_snapshot.json` e `request.json`
- `market_history_decision.csv` e `market_history_trend.csv`

Previsões antigas nunca são reestimadas com o modelo de hoje.

## Critérios para avançar

Depois do teste local, executar o primeiro experimento Binance 4 anos
e enviar o ZIP. Avaliar por horizonte:

1. Se ROC-AUC está consistentemente acima de 0,5 em diferentes folds.
2. Se Brier Skill é positivo versus frequência do treino.
3. Se MAE/RMSE superam retorno zero e retorno médio do treino.
4. Se existe estabilidade temporal e a origem/horizonte permanecem íntegros.

Somente depois de discutir esses resultados acrescentaremos **LHS/CARO**
para ajustar hiperparâmetros, guardando todos os candidatos e preservando
um período final não utilizado para selecionar configurações.

**Não há** política de entrada ou saída BTC/CASH nesta etapa. A criação
das políticas e a comparação de capital, custos, Sharpe, drawdown e
proximidade dos topos ficam para a etapa posterior.

## Testes de validação local

```bash
git fetch origin
git switch --track origin/feature/btc-direction-return-v1-20261107
poetry run python -m pytest tests/test_direct_forecast.py -v
poetry run python -m pytest -v
```
