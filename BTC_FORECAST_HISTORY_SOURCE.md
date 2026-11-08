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
- Se a série principal apresentar até 12 horários UTC ausentes, a API
  tenta consultar os ZIPs **diários oficiais da própria Binance**, com
  verificação de SHA-256, somente para os horários ausentes. Cada ZIP
  complementar fica registrado no manifesto da execução.
- Se o candle também não estiver disponível no arquivo diário, ou
  se houver mais de 12 ausências, o backtest falha informando os
  timestamps UTC faltantes e as tentativas realizadas. Nunca reduz
  automaticamente a janela nem substitui preços por interpolação,
  candles artificiais ou dados da MEXC.
- A contagem solicitada deve ser alcançada **exatamente**, e o período
  recuperado deve ter intervalos cronológicos consecutivos.
  Uma ausência persistente é uma limitação real da fonte, e não
  justificativa para fabricar uma observação.
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

## Falha "Requested 70378, only 70376"

Uma coleta real de 08/11/2026 encontrou dois candles a menos do que o
solicitado. O erro anterior não informava se a ausência estava nas
bordas da janela ou dentro da série. Agora a coleta:

1. Calcula todos os horários UTC esperados e compara com os recebidos.
2. Tenta recuperar até 12 lacunas nos arquivos diários do mesmo
   mercado Binance Spot, autenticando os ZIPs por SHA-256.
3. Registra no manifesto arquivos complementares, hashes e quantidade
   de candles recuperados.
4. Falha indicando os horários ausentes caso não consiga a série completa.

Essa correção preserva o rigor da pesquisa. Ela não garante que candles
nunca negociados ou não publicados pela corretora possam ser recuperados.


## Interrupção documentada em 24/03/2023

A Binance publicou um comunicado oficial confirmando a suspensão de todas
as negociações Spot em 24/03/2023 e o retorno às 14:00 UTC:

https://www.binance.com/en/support/announcement/detail/813a31506e9f478ea8c1058b425df87a

O arquivo mensal e o ZIP diário de BTCUSDT 30min não contêm os candles
de 13:00 e 13:30 UTC. Isso está consistente com a interrupção.

Esta única exceção documentada, e somente para BTCUSDT 30min na Binance
Spot, permite um backtest com 70.376 registros reais em vez dos 70.378
intervalos teóricos. O horário de cada candle ausente continua no
\`market_interruptions.json\` e no \`history_coverage.json\`.

Os indicadores de decisão são reiniciados depois da lacuna. Os rótulos
que exigiriam observar preços através da suspensão não são usados
no treinamento ou na avaliação. Nenhum candle artificial é criado.
Todas as demais lacunas permanecem bloqueadas.

\`FULL_WITH_DOCUMENTED_HALT\` descreve uma janela histórica integral,
exceto a interrupção conhecida e contabilizada; não significa 70.378
candles reais.
