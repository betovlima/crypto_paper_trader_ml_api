# BTC Forecast: coleta histórica de quatro anos

## Por que esta fonte foi adicionada

A auditoria MEXC de novembro de 2026 mostrou que um backtest solicitado
para quatro anos de candles de 30 minutos recebeu apenas 17.280 candles,
aproximadamente 360 dias. O limite cadastrado `market.max_candles=100000`
não explicou essa limitação. Não foi demonstrada a causa externa exata.

A fonte opcional `BINANCE_SPOT_ARCHIVE` usa os arquivos Spot públicos
e oficiais da Binance para **pesquisa histórica**, preservando a fonte
de todos os candles de uma execução. A coleta não mistura Binance e MEXC.

## Como escolher pelo Front

Na aba `ML Backtest`, clique em `Nova execução` e selecione:

- Fonte histórica: `Binance Spot (pesquisa histórica)`
- Ativo: `BTCUSDT`
- Período: `4 anos`
- Candle de decisão: `30min`
- Candle de tendência: `4hour`

O padrão continua `MEXC`. `backtest.default_history_source` é
configurável no banco pela tela **Parâmetros**, não em `.env`.
O formulário registra a origem no snapshot da execução.
Não são habilitadas ordens reais.

## Regras de recuperação

- URL fixa: `https://data.binance.vision/data/spot`.
- Arquivos mensais para meses completos e arquivos diários para o mês recente.
- A coleta exclui o dia UTC atual e o dia imediatamente anterior, porque
  arquivos diários precisam ser publicados antes do backtest. O relatório
  registra a janela efetiva, inclusive esse atraso.
- SHA-256 de **cada ZIP** é validado usando o respectivo arquivo
  `.CHECKSUM` oficial. Hash, URL e número de linhas ficam em
  `model_metadata.json` e `manifest.json`.
- A mudança de timestamps Spot para microssegundos em 01/01/2025 é
  tratada por candle; timestamps internos retornam UTC e milissegundos.
- Candles faltantes e arquivos indisponíveis geram erro explícito.
  Não há interpolação, reconstrução ou complemento automático com MEXC.
- `market.max_candles` continua sendo respeitado.
- A exportação inclui candles brutos, folds, probabilidades OOS,
  patrimônio, operações, parâmetros e proveniência da origem.

## Interpretação dos resultados

Os preços Spot da Binance podem diferir dos preços MEXC. Os custos
de negociação continuam os configurados para a **simulação** e não
devem ser tratados como taxas efetivamente verificadas da Binance.
Não compare patrimônio de execuções Binance e MEXC como se fossem
operadas no mesmo mercado com a mesma liquidez.

Não foi treinado um novo modelo de previsão de direção de BTC nesta etapa.
O XGBoost atual continua classificando atingimento de alvos/stops.
Hiperparametrização e CARO entram após validarmos os dados históricos,
os folds e os baselines.

## Testes locais

```bash
git switch feature/btc-forecast-four-year-history-20261107
poetry run python -m pytest tests/test_binance_archive.py -v
poetry run python -m pytest -v
```

Os testes de coleta usam tráfego HTTP simulado, sem baixar arquivos
de qualquer corretora. O download real precisa ser verificado executando
um novo backtest e exportando um ZIP com `history_source` e
`market_archive_checksums`.
