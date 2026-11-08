# BTC Direction & Return V2 | Operação integral pelo Front

## Objetivo comercial

O produto final deverá capitalizar USDT com apenas duas transições:
**CASH → BTC** (compra) e **BTC → CASH** (venda), com taxas, slippage,
controle de risco e reinvestimento do capital efetivo.

A V2 atual investiga a qualidade preditiva e a regressão. **Ainda não
executa operações financeiras BTC/CASH com estes novos modelos.**
Lucro por ciclo, patrimônio e operações paper pertencem à integração
posterior do simulador financeiro.

## Teste 1: comparação controlada V1 × V2 inteiramente pelo Front

1. Abra a aba **BTC Forecast**.
2. Selecione uma execução original **BTC Direction & Return**, concluída,
   com quatro anos da Binance e auditoria armazenada na mesma ML API.
3. Clique em **Reexecutar e comparar V1 × V2**.
4. Acompanhe a nova execução e seu progresso na própria tela.
5. Abra o resultado concluído, que passa a apresentar a seção
   **Comparação V1 × V2** com ROC-AUC, Brier Skill, MAE, RMSE e
   previsões distintas por fold.

A API lê o ZIP original diretamente de ml_backtest_audits no banco,
reutiliza os candles arquivados e aplica os novos parâmetros de regressão
configurados no painel. Não baixa preços nem usa arquivos locais.
O resultado é persistido como uma nova execução, com referência
replay_parent_job_id à execução original e replay_comparison no resultado.

A comparação exige igualdade dos históricos brutos, dos rótulos futuros
e das fronteiras de treino, validação, calibração e teste. Caso contrário,
a execução apresenta um erro na tela, sem inventar comparabilidade.

Se o ZIP não estiver persistido na ML API conectada, selecione outra
execução concluída com auditoria armazenada. O sistema não procura
automaticamente arquivos no computador.

**Não é necessário executar scripts, informar caminhos, abrir terminal
nem exportar o ZIP para iniciar o replay.** O botão de exportação continua
disponível apenas para auditoria posterior.

## Teste 2: nova execução completa pelo Front

Em **ML Backtest → Nova execução**, escolha:
- Tipo: **BTC Direction & Return**
- Fonte: **Binance Spot Archive**
- Ativo: BTCUSDT
- Período solicitado: quatro anos
- Decisão: 30 minutos
- Tendência: quatro horas

Na aba Parâmetros, os valores padrão V2 são escala 10.000,
gamma do regressor 0, penalidade L1 0, 350 árvores máximas
e 3 folds. Todos são mantidos no banco, com snapshot imutável
por execução, sem configurações de trading em variáveis .env.

Nesta segunda execução, a janela de preços poderá estar mais recente.
**Não interprete diferenças da nova coleta como melhora da V2.**
Para comparar de forma controlada, use o Teste 1.

## O que observar na interface

- O regressor de 2h deixou de prever um único retorno em cada fold?
- Quantas divisões reais ocorreram nas árvores?
- O retorno previsto variou mais sem piorar MAE/RMSE?
- A calibração melhorou o Brier em relação às probabilidades brutas?
- As métricas de classificação são consistentes nos três folds?
- Qual é a incerteza estimada por blocos temporais?

O número de valores distintos por fold é diagnóstico de modelagem, não
um sinal suficiente de lucro.

## O que falta ao produto BTC/CASH

A próxima etapa é a máquina de estados CASH ↔ BTC, com:
execução após o sinal conhecido, taxas, slippage, reinvestimento
integral ou parametrizado, contabilização de cada ciclo, lucro líquido,
capital final, CAGR, drawdown, Sharpe/Sortino, exposição e Buy and Hold.

Depois haverá paper trading, reconciliação de posições, recuperação
de falhas, limites operacionais e isolamento por usuário.
O objetivo final é lucro líquido ajustado ao risco, não apenas ROC-AUC.
