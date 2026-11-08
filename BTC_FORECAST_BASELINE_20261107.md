# BTC Forecast: baseline antes das correções temporais

A execução original foi fornecida como ZIP pelo responsável pelo projeto.
**Não foi reexecutada nem substituída.** Os resultados abaixo derivam de
`btc-forecast-audit-45793673-bd81-4e86-b573-3a952db90df0.zip`.

## Execução original (metodologia anterior)

| Campo | Valor |
| --- | --- |
| Identificador | `45793673-bd81-4e86-b573-3a952db90df0` |
| Mercado | BTCUSDT |
| Candles de decisão | 30 minutos |
| Candles de tendência | 4 horas |
| Amostra fora do treino (OOS) | 2026-04-25T00:00:00Z até 2026-11-07T18:00:00Z |
| Candles OOS | 9.445 |
| Folds OOS | 6 |
| Patrimônio inicial | 1.000,00 USDT |
| Patrimônio final da estratégia | 1.000,00 USDT |
| Operações | 0 |
| Exposição ao BTC | 0% |
| Retorno da estratégia | 0% |
| Retorno do Buy and Hold anterior | aproximadamente -1,96%, **sem paridade de custos** |
| Brier de entrada | 0,1916 |
| Brier de saída | 0,2513 |
| PR-AUC de entrada | 0,2969 |
| PR-AUC de saída | 0,4637 |
| Limiar de compra | 0,65 |

Essa execução permaneceu em CASH e **não comprovou capacidade de
prever o movimento ou operar com lucro**. O relatório anterior comparou
contra Buy and Hold sem aplicar os mesmos custos de negociação.

## Mudanças metodológicas em revisão

A branch `fix/btc-forecast-temporal-validation-20261107` corrige:

1. Alinhamento dos indicadores de 4h/1d ao **fechamento** do candle, e
   não ao seu timestamp de abertura.
2. Separação por um horizonte de rótulos entre treino, calibração e teste.
3. Registro das fronteiras de cada fold e das probabilidades baselines
   extraídas somente do treino.
4. Simulação Buy and Hold com taxa e slippage equivalentes.
5. Diagnóstico explícito dos filtros que impedem operações e do número
   efetivo de entradas e saídas.
6. Relatório de histórico disponível, lacunas e diferença para os anos
   pedidos no backtest.
7. ZIP de auditoria v2 contendo candles brutos, folds e diagnósticos.

**As métricas das duas versões não devem ser tratadas como a mesma experiência.**
O novo alinhamento temporal muda atributos e, portanto, as probabilidades do
modelo. Não há garantia de melhora de retorno ou de taxa de acerto.

O próximo experimento deverá exportar novo ZIP e registrar a janela
de dados, a distribuição das probabilidades, o número de operações,
o Brier relativo ao baseline cronológico e as métricas financeiras
com custos equivalentes.

## Pendência metodológica específica

Os rótulos de atingimento de alvo/stop continuam diferentes dos
limites de risco e realização empregados pela estratégia de negociação.
Não os alteramos silenciosamente porque isso mudaria a pergunta
que o modelo atual responde. A próxima etapa desenvolverá **modelos
separados de direção e retorno futuro**, seguidos de uma política BTC/CASH
com parâmetros próprios, validada fora da amostra. Só depois deverão
ser otimizados com LHS/CARO.
