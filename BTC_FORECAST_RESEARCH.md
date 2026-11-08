# BTC Forecast: plano de experimentos e rastreabilidade

Este documento orienta as implementações incrementais do BTC Forecast nas três aplicações
(Crypto Paper Trader API, ML API e Front). Não contém resultados que ainda não foram executados.

## Etapa 1: linha do tempo de posição efetiva (implementada nesta branch)

- Não altera a estratégia XGBoost atual, nem a simulação financeira original.
- Expõe `rotation_timeline` em novos resultados de backtests ML.
- Cada ponto registra o fechamento do BTC, a posição BTC/CASH *após* o candle,
  o patrimônio, o benchmark e as execuções simuladas daquele candle.
- Em caso de compra e stop no mesmo candle, os dois eventos são preservados.
- O encerramento no fim do teste aparece como saída efetiva.
- A amostragem preserva todos os candles com execução, mesmo que ultrapasse o
  número normal de pontos usado para desenhar o gráfico.
- A distância atual `distance_to_peak = close / max(close até t) - 1`
  utiliza o maior **fechamento acumulado**, e não máximas intrabar ou topos
  locais identificados posteriormente.
- O registro analítico do candle ocorre **após o seu fechamento**. Sua distância
  ao topo não pode ser usada para justificar uma ordem executada na abertura.

A versão nova da ML API é necessária para gravar esse campo. Resultados de
execuções anteriores não têm `rotation_timeline` e devem permanecer assim.

## Etapa 2: previsões fora da amostra

- Treinar direção, retorno percentual e faixas preditivas para 2, 6 e 24 horas.
- Definir rótulos após o fechamento do candle de origem e com o horizonte futuro
  integralmente separado dos exemplos de treinamento.
- Garantir que indicadores de 4h/1d só sejam usados após o respectivo fechamento.
- Registrar timestamp UTC, horizonte, features disponíveis, versão do modelo,
  probabilidades e resultado realizado, sem reconstruir o passado com modelos novos.
- Comparar contra baselines: frequência histórica, movimento nulo e XGBoost atual.
- Métricas: Brier, Log Loss, erro de retorno, calibração, cobertura das faixas e
  estabilidade por período.

## Etapa 3: hiperparametrização LHS/CARO

- Usar LHS para exploração inicial, seguido por seleção adaptativa CARO.
- Separar treinamento, calibração, otimização e avaliação final temporalmente.
- Guardar identificador, parâmetros, semente, dados e folds de cada candidato.
- Comparar modelo padrão, hiperparametrização isolada e CARO.
- O teste final não deve participar da seleção dos parâmetros.

## Etapa 4: decisões BTC/CASH e proximidade dos topos locais

- Selecionar uma política de alocação a partir de previsões fora da amostra.
- Otimizar custos, proteção, exposição, drawdown e estabilidade por período.
- Comparar patrimônio líquido, Buy and Hold, Sharpe, Sortino, MaxDD e tempo em CASH.
- Analisar retrospectivamente topos *locais* definidos por janela documentada,
  separando-os do maior fechamento acumulado de cada instante.
- Identificar cada entrada/saída e a distância do preço de execução ao topo
  local ex post. Essa informação só serve para auditoria, nunca para treinamento
  ou decisões que ainda não teriam acesso ao futuro.
- Champion/Challenger com aprovação apenas após desempenho temporal independente.

## Registro de cada experimento

Para cada rodada, registrar ao menos:

| Campo | Exemplo de conteúdo |
| --- | --- |
| ID da execução | ID persistido pela ML API |
| Estratégia/modelo | Versão efetivamente executada |
| Período de dados e folds | Datas UTC e fronteiras de treino/teste |
| Configuração | Snapshot imutável dos parâmetros |
| Horizonte(s) | Candles e horas |
| Baselines | Versões e respectivas métricas |
| Qualidade preditiva | Métricas OOS, calibração e intervalos |
| Resultado financeiro | Capital, custos, Sharpe, MaxDD e exposição |
| Rotações | Entradas/saídas efetivas; período em CASH |
| Decisão | Aprovado, rejeitado ou inconclusivo, com justificativa |

A tabela deve ser preenchida com resultados reais, não com números demonstrativos.
Não alterar backtests históricos após a execução.
